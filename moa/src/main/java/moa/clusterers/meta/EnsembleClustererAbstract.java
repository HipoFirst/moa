package moa.clusterers.meta;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import com.github.javacliparser.FileOption;
import com.google.gson.Gson;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;
import moa.options.ClassOption;
import moa.streams.clustering.RandomRBFGeneratorEvents;
import moa.tasks.TaskMonitor;

// Classes to read json settings
class ParameterSettings {
	public String parameter;
	public double value;
	public double min;
	public double max;
	public String type;
	public double std;
}

class AlgorithmSettings {
	public String algorithm;
	public ParameterSettings[] parameters;
}

class GeneralSettings {
	public int windowSize;
	public int ensembleSize;
	public AlgorithmSettings[] algorithms;
}

public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int instancesSeen;
	protected Clusterer[] ensemble;
	int bestModel;
	ArrayList<DataPoint> windowPoints;
	AdaptiveRandomForestRegressor ARFreg;
	GeneralSettings settings;

	public FileOption fileOption = new FileOption("ConfigurationFile", 'f', "Configuration file in json format.",
			"settings.json", ".json", false);

	public void init() {
		this.fileOption.getFile();
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return null;
	}

	@Override
	public Clustering getClusteringResult() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void resetLearningImpl() {

		this.instancesSeen = 0;
		this.bestModel = 0;
		this.windowPoints = new ArrayList<DataPoint>(this.settings.windowSize);

		this.ARFreg = new AdaptiveRandomForestRegressor(); // create regressor
		this.ARFreg.prepareForUse();

		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i].resetLearning();
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		DataPoint point = new DataPoint(inst, instancesSeen); // create data points from instance
		this.windowPoints.add(point); // remember points of the current window
		this.instancesSeen++;

		// train all models
		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i].trainOnInstance(inst);
		}

		// every windowSize we update the configurations
		if (this.instancesSeen % this.settings.windowSize == 0) {
			System.out.println("\n-------------- Processed " + instancesSeen + " Instances --------------");
			updateConfiguration();
			windowPoints.clear(); // flush the current window
		}
	}

	protected void updateConfiguration() {
		// init evaluation measure
		SilhouetteCoefficient silh = new SilhouetteCoefficient();

		// parameter settings are the attributes, class label is the performance // TODO init for every algorithm separately?
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int j = 0; j < this.settings.algorithms[0].parameters.length; j++) {
			attributes.add(new Attribute("att" + (j + 1)));
		}
//		attributes.add(new Attribute("class1"));

		evaluatePerformance(silh, attributes);
		
		System.out.println("Clusterer " + this.bestModel + " is the active clusterer.");


		predictConfiguration(silh, attributes);
	}
	
	protected void evaluatePerformance(SilhouetteCoefficient silh, ArrayList<Attribute> attributes) {
		double maxVal = -1 * Double.MAX_VALUE;
		for (int i = 0; i < this.ensemble.length; i++) {
			// get current macro clusters
			Clustering result = this.ensemble[i].getClusteringResult();

			// evaluate clustering using silhouette width
			silh.evaluateClustering(result, null, windowPoints);
			double performance = silh.getLastValue(0);
			System.out.println(this.ensemble[i].getCLICreationString(Clusterer.class) + ":\t => \t Silhouette: " + performance);
//				System.out.println(ClassOption.stripPackagePrefix(this.ensemble[i].getClass().getName(), Clusterer.class)); // print class
//				System.out.println(this.ensemble[i].getOptions().getAsCLIString()); // print non-default options

			// find best clustering result
			if (performance > maxVal) {
				maxVal = performance;
				this.bestModel = i; // the clusterer with the best result becomes the active one
			}
			
			// create new instance based on settings and performance to train regressor
			double[] params = new double[this.settings.algorithms[i].parameters.length+1];
			for(int j=0; j<this.settings.algorithms[i].parameters.length; j++) {
				params[j] = this.settings.algorithms[i].parameters[j].value;
			}
			params[params.length-1] = performance;
			Instance inst = new DenseInstance(1.0, params);

			// add header to dataset
			Instances dataset = new Instances(null, attributes, 0);
			dataset.setClassIndex(dataset.numAttributes());

			inst.setDataset(dataset);

			// train adaptive random forest regressor based on performance of model
			this.ARFreg.trainOnInstanceImpl(inst);
		}
	}

	// predict performance of new configuration
	protected void predictConfiguration(SilhouetteCoefficient silh, ArrayList<Attribute> attributes) {

// Integer and ordinal: round(rtnorm(1, mean + 0.5, stdDev, lowerBound, upperBound + 1) - 0.5)
// real: round(rtnorm(1, mean, stdDev, lowerBound, upperBound), digits)
// categorical: sample(x = possibleValues, size = 1, prob = probVector)

// std: newProbVector <- probVector[1] * ((1 / nbNewConfigurations)^(1 / parameters$nbVariable))
		
		// sample parent configuration
		ArrayList<Double> silhs = silh.getAllValues(0);
		int parentIdx = sampleProportionally(silhs);
		System.out.println("Selected Configuration " + parentIdx + " as parent: " + this.ensemble[parentIdx].getCLICreationString(Clusterer.class));
		
		// sample new configuration using truncated normal distribution 
		double[] vals = new double[this.settings.algorithms[parentIdx].parameters.length];
		for(int i=0; i<this.settings.algorithms[parentIdx].parameters.length; i++) {
			
			if(this.settings.algorithms[parentIdx].parameters[i].type.equals("numeric")) {
				double mean = this.settings.algorithms[parentIdx].parameters[i].value;
				double std = this.settings.algorithms[parentIdx].parameters[i].std;
				double lb = this.settings.algorithms[parentIdx].parameters[i].min;
				double ub = this.settings.algorithms[parentIdx].parameters[i].max;
				System.out.println("Sample based on mean: " + mean + ", std: " + std + ", lb: " + lb + ", ub: " + ub);
				TruncatedNormal trncnormal = new TruncatedNormal(mean, std, lb, ub);
				vals[i] = trncnormal.sample();
			}
			
		}
		Instance newInst = new DenseInstance(1.0, vals);
		Instances newDataset = new Instances(null, attributes, 0);
		newDataset.setClassIndex(newDataset.numAttributes());
		newInst.setDataset(newDataset);

		double prediction = this.ARFreg.getVotesForInstance(newInst)[0];
		System.out.println("Predict " + Arrays.toString(vals) + ":\t => \t Silhouette: " + prediction);
		
		// if we found a promising solution
		if(prediction > silh.getMinValue(0)){
			// replace by sampling proportionally
			int replaceIdx = sampleProportionally(silhs);
			System.out.println("New Configuration appears to be promising! Replace configuration: " + replaceIdx);
			
			// store new config in settings
			for(int j=0; j<vals.length-1; j++) {
				this.settings.algorithms[replaceIdx].parameters[j].value = vals[j];
			}
				
			// construct CLI string
			StringBuilder commandLine = new StringBuilder();
			commandLine.append(this.settings.algorithms[replaceIdx].algorithm); // add algorithm class
			for (ParameterSettings option : this.settings.algorithms[replaceIdx].parameters) {
				commandLine.append(" ");
				commandLine.append("-" + option.parameter); // as well as all parameters
				commandLine.append(" " + option.value);
			}
			System.out.println("Initialise: " + commandLine.toString());

			// create new clusterer from CLI string
			ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
			this.ensemble[replaceIdx] = (Clusterer) opt.materializeObject(null, null);
			this.ensemble[replaceIdx].prepareForUse();
		}
	}
	
	protected int sampleProportionally(ArrayList<Double> values) {
        double completeWeight = 0.0;
        for (Double value : values)
            completeWeight += value;
        
        double r = Math.random() * completeWeight;
        double countWeight = 0.0;
        for (int j=0; j<values.size(); j++) {
            countWeight += values.get(j);
            if (countWeight >= r)
                return j;
        }
        throw new RuntimeException("Sampling failed");
    }
	
	

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
	}

	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

		// read settings from json
		try {
			BufferedReader bufferedReader = new BufferedReader(new FileReader(fileOption.getValue()));
			Gson gson = new Gson();
			this.settings = gson.fromJson(bufferedReader, GeneralSettings.class);

			this.ensemble = new Clusterer[this.settings.algorithms.length];
			// for all algorithms
			for (int i = 0; i < this.settings.algorithms.length; i++) {

				// construct CLI string
				StringBuilder commandLine = new StringBuilder();
				commandLine.append(this.settings.algorithms[i].algorithm); // add algorithm class
				for (ParameterSettings option : this.settings.algorithms[i].parameters) {
					commandLine.append(" ");
					commandLine.append("-" + option.parameter); // as well as all parameters
					commandLine.append(" " + option.value);
				}
				System.out.println("Initialise: " + commandLine.toString());

				// create new clusterer from CLI string
				ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
				this.ensemble[i] = (Clusterer) opt.materializeObject(monitor, repository);
			}
			
		} catch (

		FileNotFoundException e) {
			e.printStackTrace();
		}
		super.prepareForUseImpl(monitor, repository);

	}

	public static void main(String[] args) {
		EnsembleClustererBlast algorithm = new EnsembleClustererBlast();
		RandomRBFGeneratorEvents stream = new RandomRBFGeneratorEvents();
		stream.prepareForUse();
		algorithm.prepareForUse();
		for (int i = 0; i < 100000; i++) {
			Instance inst = stream.nextInstance().getData();
			algorithm.trainOnInstanceImpl(inst);
		}
		algorithm.getClusteringResult();

//		System.out.println("-------------");
//		
//		EnsembleClusterer algorithm2 = new EnsembleClusterer();
//		RandomRBFGeneratorEvents stream2 = new RandomRBFGeneratorEvents();
//		stream2.prepareForUse();
//		algorithm2.prepareForUse();
//		for(int i=0; i<100000; i++) {
//			Instance inst = stream2.nextInstance().getData();
//			algorithm2.trainOnInstanceImpl(inst);
//		}
//		algorithm2.getClusteringResult();

	}

}
