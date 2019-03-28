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


// these classes are initialised by gson and contain the starting configurations
// we use these configurations to initialise the ensemble using the same classes
// This class contains the individual parameter settings (such as limits and current value)
class ParameterSettings {
	public String parameter;
	public double value;
	public double min;
	public double max;
	public String type;
	public double std;

	// since a custom constructor is not called by gson, we provide an init method
	public void prepareForUse(){
		this.std = (this.max - this.min) / 2;
	}

	// copy constructor
	public ParameterSettings(ParameterSettings x){
		this.parameter = x.parameter;
		this.value = x.value;
		this.min = x.min;
		this.max = x.max;
		this.type = x.type;
		this.std = x.std;
	}
}

// This class contains the settings of an algorithm (such as name and the actual clusterer object) as well as an array of Parameter Settings
class AlgorithmSettings {
	public String algorithm;
	public ParameterSettings[] parameters;
	public Clusterer clusterer;

	// since a custom constructor is not called by gson, we provide an init method
	public void prepareForUse(){
		// initialise a new algorithm using the Command Line Interface (CLI)
		// construct CLI string from settings, e.g. denstream.WithDBSCAN -e 0.08 -b 0.3
		StringBuilder commandLine = new StringBuilder();
		commandLine.append(this.algorithm); // first the algorithm class
		for (ParameterSettings option : this.parameters) {
			commandLine.append(" ");
			commandLine.append("-" + option.parameter); // then the parameter
			commandLine.append(" " + option.value); // and its value
		}
		System.out.println("Initialise: " + commandLine.toString());

		// create new clusterer from CLI string
		ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
		this.clusterer = (Clusterer) opt.materializeObject(null, null);
		this.clusterer.prepareForUse();

		for(ParameterSettings parameter : this.parameters){
			parameter.prepareForUse();
		}
	}

	public AlgorithmSettings(AlgorithmSettings x){
		this.algorithm = x.algorithm;
		this.parameters = new ParameterSettings[x.parameters.length];
		for(int i=0; i < x.parameters.length; i++){
			this.parameters[i] = new ParameterSettings(x.parameters[i]);
		}
	}
}

// This contains the general settings (such as the max ensemble size) as well as an array of Algorithm Settings
class GeneralSettings {
	public int windowSize;
	public int ensembleSize;
	public AlgorithmSettings[] algorithms;
}


public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int instancesSeen;
	int currentEnsembleSize;
	int bestModel;
	ArrayList<AlgorithmSettings> ensemble;
	ArrayList<DataPoint> windowPoints;
	AdaptiveRandomForestRegressor ARFreg;
	GeneralSettings settings;

	// the file option dialogue in the UI
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

		// create or reset regressor
		this.ARFreg = new AdaptiveRandomForestRegressor(); 
		this.ARFreg.prepareForUse();

		// reset individual clusterers
		for (int i = 0; i < this.ensemble.size(); i++) {
			this.ensemble.get(i).clusterer.resetLearning();
		}
	}


	@Override
	public void trainOnInstanceImpl(Instance inst) {
		DataPoint point = new DataPoint(inst, instancesSeen); // create data points from instance
		this.windowPoints.add(point); // remember points of the current window
		this.instancesSeen++;

		// train all models with the instance
		for (int i = 0; i < this.ensemble.size(); i++) {
			this.ensemble.get(i).clusterer.trainOnInstance(inst);
		}

		// every windowSize we update the configurations
		if (this.instancesSeen % this.settings.windowSize == 0) {
			System.out.println("\n-------------- Processed " + instancesSeen + " Instances --------------");
			updateConfiguration(); // update configuration
			windowPoints.clear(); // flush the current window
		}
	}


	protected void updateConfiguration() {
		// init evaluation measure (silhouette for now)
		SilhouetteCoefficient silh = new SilhouetteCoefficient();

		// parameter settings are the attributes, class label is the performance // TODO init for every algorithm separately?
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int j = 0; j < this.settings.algorithms[0].parameters.length; j++) {
			attributes.add(new Attribute("att" + (j + 1)));
		}
		// attributes.add(new Attribute("class1"));

		// train the random forest regressor based on the configuration performance
		// and find the best performing algorithm
		evaluatePerformance(silh, attributes);
		
		System.out.println("Clusterer " + this.bestModel + " is the active clusterer.");

		// generate a new configuration and predict its performance using the random forest regressor
		predictConfiguration(silh, attributes);
	}
	

	protected void evaluatePerformance(SilhouetteCoefficient silh, ArrayList<Attribute> attributes) {
		double maxVal = -1 * Double.MAX_VALUE;
		for (int i = 0; i < this.ensemble.size(); i++) {
			// get macro clusters of this clusterer
			Clustering result = this.ensemble.get(i).clusterer.getClusteringResult();

			// evaluate clustering using silhouette width
			silh.evaluateClustering(result, null, windowPoints);
			double performance = silh.getLastValue(0);
			System.out.println(i + ") " + this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class) + ":\t => \t Silhouette: " + performance);
			// System.out.println(ClassOption.stripPackagePrefix(this.ensemble[i].getClass().getName(), Clusterer.class)); // print class
			// System.out.println(this.ensemble[i].getOptions().getAsCLIString()); // print non-default options

			// find best clustering result among all algorithms
			if (performance > maxVal) {
				maxVal = performance;
				this.bestModel = i; // the clusterer with the best result becomes the active one
			}
			
			// create new training instance based to train regressor
			// features are the algorithm and its configuration, the class is its performance in the last window
			double[] params = new double[this.ensemble.get(i).parameters.length+1];
			for(int j=0; j<this.ensemble.get(i).parameters.length; j++) {
				params[j] = this.ensemble.get(i).parameters[j].value; // add configuration as features
			}
			params[params.length-1] = performance; // add performance as class
			Instance inst = new DenseInstance(1.0, params);

			// add header to dataset (same as before, TODO: no attribute for class, not sure if problem)
			Instances dataset = new Instances(null, attributes, 0);
			dataset.setClassIndex(dataset.numAttributes()); // set class index to our performance feature
			inst.setDataset(dataset);
			
			// train adaptive random forest regressor based on performance of model
			this.ARFreg.trainOnInstanceImpl(inst);
		}
	}


	// predict performance of new configuration
	protected void predictConfiguration(SilhouetteCoefficient silh, ArrayList<Attribute> attributes) {

		// irace sampling of new configurations:
		// Integer and ordinal: round(rtnorm(1, mean + 0.5, stdDev, lowerBound, upperBound + 1) - 0.5)
		// real: round(rtnorm(1, mean, stdDev, lowerBound, upperBound), digits)
		// categorical: sample(x = possibleValues, size = 1, prob = probVector)

		// irace adaption of sampling parameters:
		// mean taken from parent
		// std: newProbVector <- probVector[1] * ((1 / nbNewConfigurations)^(1 / parameters$nbVariable))
		
		// sample a parent configuration proportionally to its performance from the ensemble
		ArrayList<Double> silhs = silh.getAllValues(0);
		int parentIdx = sampleProportionally(silhs);
		System.out.println("Selected Configuration " + parentIdx + " as parent: " + this.ensemble.get(parentIdx).clusterer.getCLICreationString(Clusterer.class));
		
		// sample new configuration from the parent
		double[] vals = new double[this.ensemble.get(parentIdx).parameters.length];
		for(int i=0; i<this.ensemble.get(parentIdx).parameters.length; i++) {
			
			// for numeric features using truncated normal distribution 
			if(this.ensemble.get(parentIdx).parameters[i].type.equals("numeric")) {
				double mean = this.ensemble.get(parentIdx).parameters[i].value;
				double std = this.ensemble.get(parentIdx).parameters[i].std;
				double lb = this.ensemble.get(parentIdx).parameters[i].min;
				double ub = this.ensemble.get(parentIdx).parameters[i].max;
				TruncatedNormal trncnormal = new TruncatedNormal(mean, std, lb, ub);
				vals[i] = trncnormal.sample();
				System.out.println("Sample new configuration for parameter " + i + " with mean: " + mean + ", std: " + std + ", lb: " + lb + ", ub: " + ub + ":" + vals[i]);
			} else{
				throw new RuntimeException("Only numeric features implemented so far.");
			}	
		}

		Instance newInst = new DenseInstance(1.0, vals);
		Instances newDataset = new Instances(null, attributes, 0);
		newDataset.setClassIndex(newDataset.numAttributes());
		newInst.setDataset(newDataset);

		// predict performance of configuration
		double prediction = this.ARFreg.getVotesForInstance(newInst)[0];
		System.out.println("Predict " + Arrays.toString(vals) + ":\t => \t Silhouette: " + prediction);

		// if we still have open slots in the ensemble (not full)
		if(this.ensemble.size() < this.settings.ensembleSize){
			System.out.println("Ensemble not full. Add configuration as new algorithm.");

			// copy existing clusterer configuration but change settings
			AlgorithmSettings newConfig = new AlgorithmSettings(this.ensemble.get(parentIdx));
			for(int i=0; i<vals.length-1; i++) {
				newConfig.parameters[i].value = vals[i];
			}
			// initialise and add to ensemble
			newConfig.prepareForUse();
			this.ensemble.add(newConfig);

		} else if(prediction > silh.getMinValue(0)){
			// if the predicted performance is better than the one we have in the ensemble

			// proportionally sample a configuration that will be replaced
			int replaceIdx = sampleProportionally(silhs);
			System.out.println("Ensemble already full but new configuration is promising! Replace algorithm: " + replaceIdx);
			
			// use existing clusterer but change settings
			for(int j=0; j<vals.length-1; j++) {
				this.ensemble.get(replaceIdx).parameters[j].value = vals[j];
			}
			// and reinitialise (remains in ensemble)
			this.ensemble.get(replaceIdx).prepareForUse();
		}
	}

	
	// sample an index from a list of values, proportionally to the respective value
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

		try {
			// read settings from json
			BufferedReader bufferedReader = new BufferedReader(new FileReader(fileOption.getValue()));
			Gson gson = new Gson();
			// store settings in dedicated class structure
			this.settings = gson.fromJson(bufferedReader, GeneralSettings.class);

			// also create the ensemble which can be larger than the provided (starting) configurations
			this.ensemble = new ArrayList<AlgorithmSettings>(this.settings.ensembleSize);
			 // copy and initialise the provided starting configurations in the ensemble
			for (int i = 0; i < this.settings.algorithms.length; i++) {
				this.ensemble.add(this.settings.algorithms[i]);
				this.ensemble.get(i).prepareForUse();
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
