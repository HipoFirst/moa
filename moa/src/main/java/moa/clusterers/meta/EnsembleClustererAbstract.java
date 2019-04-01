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
class ParameterConfiguration {
	public String parameter;
	public Object value;
	public Object[] range;
	public String type;
}

// This class contains the settings of an algorithm (such as name) as well as an
// array of Parameter Settings
class AlgorithmConfiguration {
	public String algorithm;
	public ParameterConfiguration[] parameters;
}

// This contains the general settings (such as the max ensemble size) as well as
// an array of Algorithm Settings
class GeneralConfiguration {
	public int windowSize;
	public int ensembleSize;
	public int newConfigurations;
	public AlgorithmConfiguration[] algorithms;
}





class NumericalParameter {
	public String parameter;
	public double value;
	public double[] range;
	public double std;
	public Attribute attribute;
	

	public NumericalParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = (double) x.value;
		this.range = new double[x.range.length];
		for (int i = 0; i < x.range.length; i++){
			range[i] = (double) x.range[i];
		}
		this.std = (this.range[1] - this.range[0]) / 2;
		this.attribute = new Attribute(x.parameter);
	}
}

class NominalParameter {
	public String parameter;
	public String value;
	public String[] range;
	public Attribute attribute;

	public NominalParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = String.valueOf(x.value);
		this.range = new String[x.range.length];
		for (int i = 0; i < x.range.length; i++){
			range[i] = String.valueOf(x.range[i]);
		}
		this.attribute = new Attribute(x.parameter, Arrays.asList(range));

	}
}

class IntegerParameter {
	public String parameter;
	public int value;
	public int[] range;
	public Attribute attribute;
	public double std;


	public IntegerParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = (int) (double)x.value; //TODO fix casts
		this.range = new int[x.range.length];
		for (int i = 0; i < x.range.length; i++){
			range[i] = (int) (double)x.range[i];
		}
		this.std = (this.range[1] - this.range[0]) / 2;
		this.attribute = new Attribute(x.parameter);
	}
}

class Algorithm {
	public String algorithm;
	public ArrayList<NumericalParameter> numericalParameters;
	public ArrayList<NominalParameter> nominalParameters;
	public ArrayList<IntegerParameter> integerParameters;
	public Clusterer clusterer;
	public ArrayList<Attribute> attributes;

	public Algorithm(Algorithm x) {

		this.algorithm = x.algorithm;
		// TODO this is probably referencing, should be a deep copy or a new object entirely
		this.numericalParameters = x.numericalParameters; 
		this.nominalParameters = x.nominalParameters;
		this.integerParameters = x.integerParameters;
		this.attributes = x.attributes;

		// init(); // we dont initialise here because we want to manipulate the parameters first
	}

	public Algorithm(AlgorithmConfiguration x) {

		this.algorithm = x.algorithm;
		this.numericalParameters = new ArrayList<NumericalParameter>();
		this.nominalParameters = new ArrayList<NominalParameter>();
		this.integerParameters = new ArrayList<IntegerParameter>();
		this.attributes = new ArrayList<Attribute>();
		for (ParameterConfiguration param : x.parameters) {
			if (param.type.equals("numeric")) {
				NumericalParameter numParam = new NumericalParameter(param);
				this.numericalParameters.add(numParam);
				this.attributes.add(new Attribute(numParam.parameter));
			} else if (param.type.equals("integer")){
				IntegerParameter intParam = new IntegerParameter(param);
				this.integerParameters.add(intParam);
				this.attributes.add(new Attribute(intParam.parameter));
			} else if (param.type.equals("categorical")) {
				NominalParameter nomParam = new NominalParameter(param);
				this.nominalParameters.add(nomParam);
				this.attributes.add(new Attribute(nomParam.parameter, Arrays.asList(nomParam.range)));
			}
		}
		init();
	}

	public void init(){
		// initialise a new algorithm using the Command Line Interface (CLI)
		// construct CLI string from settings, e.g. denstream.WithDBSCAN -e 0.08 -b 0.3
		StringBuilder commandLine = new StringBuilder();
		commandLine.append(this.algorithm); // first the algorithm class
		for (NumericalParameter option : this.numericalParameters) {
			commandLine.append(" ");
			commandLine.append("-" + option.parameter); // then the parameter
			commandLine.append(" " + option.value); // and its value
		}
		for (IntegerParameter option : this.integerParameters) {
			commandLine.append(" ");
			commandLine.append("-" + option.parameter); // then the parameter
			commandLine.append(" " + option.value); // and its value
		}
		for (NominalParameter option : this.nominalParameters) {
			commandLine.append(" ");
			commandLine.append("-" + option.parameter); // then the parameter
			commandLine.append(" " + option.value); // and its value
		}
		System.out.println("Initialise: " + commandLine.toString());

		// create new clusterer from CLI string
		ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
		this.clusterer = (Clusterer) opt.materializeObject(null, null);
		this.clusterer.prepareForUse();
	}
}

public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int instancesSeen;
	int currentEnsembleSize;
	int bestModel;
	ArrayList<Algorithm> ensemble;
	ArrayList<DataPoint> windowPoints;
	AdaptiveRandomForestRegressor ARFreg;
	GeneralConfiguration settings;

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
		// train the random forest regressor based on the configuration performance
		// and find the best performing algorithm
		evaluatePerformance(silh);

		System.out.println("Clusterer " + this.bestModel + " is the active clusterer.");

		// generate a new configuration and predict its performance using the random
		// forest regressor
		predictConfiguration(silh);
	}

	protected void evaluatePerformance(SilhouetteCoefficient silh) {
		
		double maxVal = -1 * Double.MAX_VALUE;
		for (int i = 0; i < this.ensemble.size(); i++) {
			// get macro clusters of this clusterer
			Clustering result = this.ensemble.get(i).clusterer.getClusteringResult();

			// evaluate clustering using silhouette width
			silh.evaluateClustering(result, null, windowPoints);
			double performance = silh.getLastValue(0);
			System.out.println(i + ") " + this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class)
					+ ":\t => \t Silhouette: " + performance);
			// System.out.println(ClassOption.stripPackagePrefix(this.ensemble[i].getClass().getName(),
			// Clusterer.class)); // print class
			// System.out.println(this.ensemble[i].getOptions().getAsCLIString()); // print
			// non-default options

			// find best clustering result among all algorithms
			if (performance > maxVal) {
				maxVal = performance;
				this.bestModel = i; // the clusterer with the best result becomes the active one
			}

			// create new training instance based to train regressor
			// features are the algorithm and its configuration, the class is its
			// performance in the last window
			double[] params = new double[this.ensemble.get(i).attributes.size() + 1];
			int pos = 0;
			for (NumericalParameter param : this.ensemble.get(i).numericalParameters) {
				params[pos++] = param.value; // add configuration as features
			}
			for (IntegerParameter param : this.ensemble.get(i).integerParameters) {
				params[pos++] = param.value; // add configuration as features
			}
			for (NominalParameter param : this.ensemble.get(i).nominalParameters) {
				// params[pos++] = param.value; // add configuration as features
			}

			params[params.length - 1] = performance; // add performance as class
			Instance inst = new DenseInstance(1.0, params);

			// add header to dataset TODO: do we need an attribute for the class label?
			Instances dataset = new Instances(null, this.ensemble.get(i).attributes, 0);
			dataset.setClassIndex(dataset.numAttributes()); // set class index to our performance feature
			inst.setDataset(dataset);

			// train adaptive random forest regressor based on performance of model
			this.ARFreg.trainOnInstanceImpl(inst);
		}
	}

	// predict performance of new configuration
	protected void predictConfiguration(SilhouetteCoefficient silh) {

		// irace sampling of new configurations:
		// Integer and ordinal: round(rtnorm(1, mean + 0.5, stdDev, lowerBound,
		// upperBound + 1) - 0.5)
		// real: round(rtnorm(1, mean, stdDev, lowerBound, upperBound), digits)
		// categorical: sample(x = possibleValues, size = 1, prob = probVector)

		// irace adaption of sampling parameters:
		// mean taken from parent
		// std: newProbVector <- probVector[1] * ((1 / nbNewConfigurations)^(1 /
		// parameters$nbVariable))

		// sample a parent configuration proportionally to its performance from the
		// ensemble
		ArrayList<Double> silhs = silh.getAllValues(0);

		for (int z = 0; z < this.settings.newConfigurations; z++) {
			int parentIdx = sampleProportionally(silhs);
			System.out.println("Selected Configuration " + parentIdx + " as parent: "
					+ this.ensemble.get(parentIdx).clusterer.getCLICreationString(Clusterer.class));

			// sample new configuration from the parent
			double[] vals = new double[this.ensemble.get(parentIdx).attributes.size()];
			int pos = 0;
			for (NumericalParameter param : this.ensemble.get(parentIdx).numericalParameters) {
				// for numeric features use truncated normal distribution
				double mean = param.value;
				double std = param.std;
				double lb = param.range[0];
				double ub = param.range[1];
				TruncatedNormal trncnormal = new TruncatedNormal(mean, std, lb, ub);
				vals[pos++] = trncnormal.sample();
				System.out.println("Sample new configuration for numerical parameter -" + param.parameter + " with mean: " + mean
						+ ", std: " + std + ", lb: " + lb + ", ub: " + ub + "\t=>\t -" + param.parameter + " "
						+ vals[pos - 1]);
			}
			for (IntegerParameter param : this.ensemble.get(parentIdx).integerParameters) {
				// for numeric features use truncated normal distribution
				int mean = param.value;
				double std = param.std;
				int lb = param.range[0];
				int ub = param.range[1];
				TruncatedNormal trncnormal = new TruncatedNormal(mean, std, lb, ub);
				vals[pos++] = Math.round(trncnormal.sample());
				System.out.println("Sample new configuration for integer parameter -" + param.parameter + " with mean: " + mean
						+ ", std: " + std + ", lb: " + lb + ", ub: " + ub + "\t=>\t -" + param.parameter + " "
						+ vals[pos - 1]);
			}


			Instance newInst = new DenseInstance(1.0, vals);
			Instances newDataset = new Instances(null, this.ensemble.get(parentIdx).attributes, 0);
			newDataset.setClassIndex(newDataset.numAttributes());
			newInst.setDataset(newDataset);

			double prediction = this.ARFreg.getVotesForInstance(newInst)[0];
			System.out.print("Predict " + ClassOption
					.stripPackagePrefix(this.ensemble.get(parentIdx).clusterer.getClass().getName(), Clusterer.class));
			for (int i = 0; i < this.ensemble.get(parentIdx).numericalParameters.size(); i++) {
				System.out.print(" -" + this.ensemble.get(parentIdx).numericalParameters.get(i).parameter + " " + vals[i]);
			}
			for (int i = 0; i < this.ensemble.get(parentIdx).integerParameters.size(); i++) {
				System.out.print(" -" + this.ensemble.get(parentIdx).integerParameters.get(i).parameter + " " + vals[i+this.ensemble.get(parentIdx).numericalParameters.size()]);
			}
			System.out.println("\t => \t Silhouette: " + prediction);

			// TODO if ensemble empty, we could also just fill
			if (Double.isNaN(prediction)) {
				return;
			}

			// if we still have open slots in the ensemble (not full)
			if (this.ensemble.size() < this.settings.ensembleSize) {
				System.out.println("Ensemble not full. Add configuration as new algorithm.");

				// copy existing clusterer configuration but change settings
				// TODO maybe we should init a new one instead of copying to avoid deep copy problems
				Algorithm newAlgorithm = new Algorithm(this.ensemble.get(parentIdx));
				for (int i = 0; i < newAlgorithm.numericalParameters.size(); i++) {
					NumericalParameter newParam = newAlgorithm.numericalParameters.get(i);
					newParam.value = vals[i];

					// Reduce standard deviation for next iteration
					// TODO this is not directly transferable from irace
					newParam.std = newParam.std * (Math.pow((1.0 / this.settings.newConfigurations), (1.0 / newAlgorithm.numericalParameters.size())));
				}
				for (int i = 0; i < newAlgorithm.integerParameters.size(); i++) {
					IntegerParameter newParam = newAlgorithm.integerParameters.get(i);
					newParam.value = (int) Math.round(vals[i]);
					newParam.std = newParam.std * (Math.pow((1.0 / this.settings.newConfigurations), (1.0 / newAlgorithm.numericalParameters.size())));
				}
				for (int i = 0; i < newAlgorithm.nominalParameters.size(); i++) {
					// NominalParameter newParam = newAlgorithm.nominalParameters.get(i);
					//TODO
				}

				newAlgorithm.init();

				// add to ensemble
				this.ensemble.add(newAlgorithm);

				// update current silhouettes with the prediction
				silhs.add(prediction);

			} else if (prediction > silh.getMinValue(0)) {
				// if the predicted performance is better than the one we have in the ensemble

				// proportionally sample a configuration that will be replaced
				int replaceIdx = sampleProportionally(silhs);
				System.out.println(
						"Ensemble already full but new configuration is promising! Replace algorithm: " + replaceIdx);

				// copy existing clusterer configuration but change settings
				Algorithm newAlgorithm = new Algorithm(this.ensemble.get(parentIdx));
				for (int i = 0; i < vals.length; i++) {
					newAlgorithm.numericalParameters.get(i).value = vals[i];
				}
				newAlgorithm.init();

				// replace in ensemble
				this.ensemble.set(replaceIdx, newAlgorithm);

				// update current silhouettes with the prediction
				silhs.set(replaceIdx, prediction);
			}
		}

	}

	// sample an index from a list of values, proportionally to the respective value
	protected int sampleProportionally(ArrayList<Double> values) {
		double completeWeight = 0.0;
		for (Double value : values)
			completeWeight += value;

		double r = Math.random() * completeWeight;
		double countWeight = 0.0;
		for (int j = 0; j < values.size(); j++) {
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
			this.settings = gson.fromJson(bufferedReader, GeneralConfiguration.class);

			// also create the ensemble which can be larger than the provided (starting)
			// configurations
			this.ensemble = new ArrayList<Algorithm>(this.settings.ensembleSize);
			// copy and initialise the provided starting configurations in the ensemble
			for (int i = 0; i < this.settings.algorithms.length; i++) {
				this.ensemble.add(new Algorithm(this.settings.algorithms[i]));
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
		for (int i = 0; i < 1000000; i++) {
			Instance inst = stream.nextInstance().getData();
			algorithm.trainOnInstanceImpl(inst);
		}
		algorithm.getClusteringResult();

		// System.out.println("-------------");
		//
		// EnsembleClusterer algorithm2 = new EnsembleClusterer();
		// RandomRBFGeneratorEvents stream2 = new RandomRBFGeneratorEvents();
		// stream2.prepareForUse();
		// algorithm2.prepareForUse();
		// for(int i=0; i<100000; i++) {
		// Instance inst = stream2.nextInstance().getData();
		// algorithm2.trainOnInstanceImpl(inst);
		// }
		// algorithm2.getClusteringResult();

	}

}
