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
		for (int i = 0; i < x.range.length; i++) {
			range[i] = (double) x.range[i];
		}
		this.std = (this.range[1] - this.range[0]) / 2;
		this.attribute = new Attribute(x.parameter);
	}

	public double newConfig(int newConfigs, int nbVariables) {
		// update configuration
		// for numeric features use truncated normal distribution
		TruncatedNormal trncnormal = new TruncatedNormal(this.value, this.std, this.range[0], this.range[1]);
		this.value = trncnormal.sample();

		System.out.println("Sample new configuration for numerical parameter -" + this.parameter + " with mean: "
				+ this.value + ", std: " + this.std + ", lb: " + this.range[0] + ", ub: " + this.range[1] + "\t=>\t -"
				+ this.parameter + " " + value);

		// adapt distribution
		// TODO use attributes.size() instead?
		this.std = this.std * (Math.pow((1.0 / newConfigs), (1.0 / nbVariables))); // this.settings.newConfigurations , this.numericalParameters.size()

		return this.value;
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
		this.value = (int) (double) x.value; // TODO fix casts
		this.range = new int[x.range.length];
		for (int i = 0; i < x.range.length; i++) {
			range[i] = (int) (double) x.range[i];
		}
		this.std = (this.range[1] - this.range[0]) / 2;
		this.attribute = new Attribute(x.parameter);
	}

	public double newConfig(int newConfigs, int nbVariables) {
		// update configuration
		// for integer features use truncated normal distribution
		TruncatedNormal trncnormal = new TruncatedNormal(this.value, this.std, this.range[0], this.range[1]);
		this.value = (int) Math.round(trncnormal.sample());
		System.out.println("Sample new configuration for integer parameter -" + this.parameter + " with mean: "
				+ this.value + ", std: " + this.std + ", lb: " + this.range[0] + ", ub: " + this.range[1] + "\t=>\t -"
				+ this.parameter + " " + value);

		// adapt distribution
		this.std = this.std * (Math.pow((1.0 / newConfigs), (1.0 / nbVariables)));

		return this.value;
	}
}

// TODO I dont think there are any clustering algorithms with nominal parameters
// in MOA right now
class NominalParameter {
	public String parameter;
	public String value;
	public int numericValue;
	public String[] range;
	public Attribute attribute;
	public ArrayList<Double> probabilities;

	public NominalParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = String.valueOf(x.value);
		this.range = new String[x.range.length];
		for (int i = 0; i < x.range.length; i++) {
			range[i] = String.valueOf(x.range[i]);
			if (this.range[i].equals(this.value)) {
				this.numericValue = i; // get index of init value
			}
		}
		this.attribute = new Attribute(x.parameter, Arrays.asList(range));
		this.probabilities = new ArrayList<Double>(x.range.length);
		for (int i = 0; i < x.range.length; i++) {
			this.probabilities.add(1.0 / x.range.length); // equal probabilities
		}
	}

	public double newConfig(){
		// update configuration
		this.numericValue = EnsembleClustererAbstract.sampleProportionally(this.probabilities);
		this.value = this.range[this.numericValue];

		System.out.print(
				"Sample new configuration for nominal parameter -" + this.parameter + "with probabilities");
		for (int i = 0; i < this.probabilities.size(); i++) {
			System.out.print(" " + this.probabilities.get(i));
		}
		System.out.println("\t=>\t -" + this.parameter + " " + value);

		// adapt distribution
		for (int i = 0; i < this.probabilities.size(); i++) {
			// TODO not directly transferable, (1-((iter -1) / maxIter))
			this.probabilities.set(i,
			this.probabilities.get(i) * (1.0 - ((10 - 1.0) / 100)));
		}
		this.probabilities.set(this.numericValue,
				(this.probabilities.get(this.numericValue) + ((10 - 1.0) / 100)));

		// divide by sum
		double sum = 0.0;
		for (int i = 0; i < this.probabilities.size(); i++) {
			sum += this.probabilities.get(i);
		}
		for (int i = 0; i < this.probabilities.size(); i++) {
			this.probabilities.set(i, this.probabilities.get(i) / sum);
		}

		return numericValue;
	}
}

// TODO ordinal parameter
class BooleanParameter {
	public String parameter;
	public int numericValue;
	public String value;
	public String[] range = { "false", "true" };
	public Attribute attribute;
	public ArrayList<Double> probabilities;

	public BooleanParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = String.valueOf(x.value);
		for (int i = 0; i < this.range.length; i++) {
			if (this.range[i].equals(this.value)) {
				this.numericValue = i; // get index of init value
			}
		}
		this.attribute = new Attribute(x.parameter);

		this.probabilities = new ArrayList<Double>(2);
		for (int i = 0; i < 2; i++) {
			this.probabilities.add(0.5); // equal probabilities
		}
	}

	public double newConfig(){
		// update configuration
		this.numericValue = EnsembleClustererAbstract.sampleProportionally(this.probabilities);
		this.value = this.range[this.numericValue];
		System.out.print(
				"Sample new configuration for boolean parameter -" + this.parameter + " with probabilities");
		for (int i = 0; i < this.probabilities.size(); i++) {
			System.out.print(" " + this.probabilities.get(i));
		}
		System.out.println("\t=>\t -" + this.parameter + " " + value);

		// adapt distribution
		for (int i = 0; i < this.probabilities.size(); i++) {
			// TODO not directly transferable, (1-((iter -1) / maxIter))
			this.probabilities.set(i,
			this.probabilities.get(i) * (1.0 - ((10 - 1.0) / 100)));
		}
		this.probabilities.set(this.numericValue,
				(this.probabilities.get(this.numericValue) + ((10 - 1.0) / 100)));

		// divide by sum
		double sum = 0.0;
		for (int i = 0; i < this.probabilities.size(); i++) {
			sum += this.probabilities.get(i);
		}
		for (int i = 0; i < this.probabilities.size(); i++) {
			this.probabilities.set(i, this.probabilities.get(i) / sum);
		}

		return numericValue;
	}
}


class Algorithm {
	public String algorithm;
	public ArrayList<NumericalParameter> numericalParameters;
	public ArrayList<NominalParameter> nominalParameters;
	public ArrayList<IntegerParameter> integerParameters;
	public ArrayList<BooleanParameter> booleanParameters;
	public Clusterer clusterer;
	public ArrayList<Attribute> attributes;

	public Algorithm(Algorithm x) {

		this.algorithm = x.algorithm;
		// TODO this is probably referencing, should be a deep copy or a new object
		// entirely
		this.numericalParameters = x.numericalParameters;
		this.nominalParameters = x.nominalParameters;
		this.integerParameters = x.integerParameters;
		this.booleanParameters = x.booleanParameters;
		this.attributes = x.attributes;

		// init(); // we dont initialise here because we want to manipulate the
		// parameters first
	}

	public Algorithm(AlgorithmConfiguration x) {

		this.algorithm = x.algorithm;
		this.numericalParameters = new ArrayList<NumericalParameter>();
		this.nominalParameters = new ArrayList<NominalParameter>();
		this.integerParameters = new ArrayList<IntegerParameter>();
		this.booleanParameters = new ArrayList<BooleanParameter>();

		this.attributes = new ArrayList<Attribute>();
		for (ParameterConfiguration paramConfig : x.parameters) {
			if (paramConfig.type.equals("numeric")) {
				NumericalParameter param = new NumericalParameter(paramConfig);
				this.numericalParameters.add(param);
				this.attributes.add(new Attribute(param.parameter));
			} else if (paramConfig.type.equals("integer")) {
				IntegerParameter param = new IntegerParameter(paramConfig);
				this.integerParameters.add(param);
				this.attributes.add(new Attribute(param.parameter));
			} else if (paramConfig.type.equals("nominal")) {
				NominalParameter param = new NominalParameter(paramConfig);
				this.nominalParameters.add(param);
				this.attributes.add(new Attribute(param.parameter, Arrays.asList(param.range)));
			} else if (paramConfig.type.equals("boolean")) {
				BooleanParameter param = new BooleanParameter(paramConfig);
				this.booleanParameters.add(param);
				this.attributes.add(new Attribute(param.parameter, Arrays.asList(param.range)));
			}
		}
		init();
	}

	public void init() {
		// initialise a new algorithm using the Command Line Interface (CLI)
		// construct CLI string from settings, e.g. denstream.WithDBSCAN -e 0.08 -b 0.3
		StringBuilder commandLine = new StringBuilder();
		commandLine.append(this.algorithm); // first the algorithm class
		for (NumericalParameter option : this.numericalParameters) {
			commandLine.append(" -" + option.parameter); // then the parameter
			commandLine.append(" " + option.value); // and its value
		}
		for (IntegerParameter option : this.integerParameters) {
			commandLine.append(" -" + option.parameter);
			commandLine.append(" " + option.value);
		}
		for (NominalParameter option : this.nominalParameters) {
			commandLine.append(" -" + option.parameter);
			commandLine.append(" " + option.value);
		}
		for (BooleanParameter option : this.booleanParameters) {
			// if option is set
			if (option.numericValue == 1) {
				commandLine.append(" -" + option.parameter); // only the parameter
			}
		}
		System.out.println("Initialise: " + commandLine.toString());

		// create new clusterer from CLI string
		ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
		this.clusterer = (Clusterer) opt.materializeObject(null, null);
		this.clusterer.prepareForUse();
	}


	public double[] newConfig(int newConfigs){
		// sample new configuration from the parent
		double[] vals = new double[this.attributes.size()];
		int pos = 0;
		for (NumericalParameter param : this.numericalParameters) {
			vals[pos++] = param.newConfig(newConfigs, this.attributes.size());
		}
		for (IntegerParameter param : this.integerParameters) {
			vals[pos++] = param.newConfig(newConfigs, this.attributes.size());
		}
		for (NominalParameter param : this.nominalParameters) {
			vals[pos++] = param.newConfig();
		}
		for (BooleanParameter param : this.booleanParameters) {
			vals[pos++] = param.newConfig();
		}
		return vals;
	}
}











public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int iteration;
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
				params[pos++] = param.numericValue; // add configuration as features
			}
			for (BooleanParameter param : this.ensemble.get(i).booleanParameters) {
				params[pos++] = param.numericValue; // add configuration as features
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

		// sample a parent configuration proportionally to its performance from the
		// ensemble
		ArrayList<Double> silhs = silh.getAllValues(0);

		for (int z = 0; z < this.settings.newConfigurations; z++) {

			// copy existing clusterer configuration but change settings
			// TODO maybe we should init a new one instead of copying to avoid deep copy
			// problems
			int parentIdx = EnsembleClustererAbstract.sampleProportionally(silhs);
			Algorithm newAlgorithm = new Algorithm(this.ensemble.get(parentIdx));

			// sample new configuration from the parent
			double[] vals = newAlgorithm.newConfig(this.settings.newConfigurations);
			newAlgorithm.init();

			System.out.println("Selected Configuration " + parentIdx + " as parent: " + newAlgorithm.clusterer.getCLICreationString(Clusterer.class));


			Instance newInst = new DenseInstance(1.0, vals);
			Instances newDataset = new Instances(null, newAlgorithm.attributes, 0);
			newDataset.setClassIndex(newDataset.numAttributes());
			newInst.setDataset(newDataset);

			double prediction = this.ARFreg.getVotesForInstance(newInst)[0];
			int pos = 0;
			System.out.print("Predict " + ClassOption
					.stripPackagePrefix(newAlgorithm.clusterer.getClass().getName(), Clusterer.class));
			for (int i = 0; i < newAlgorithm.numericalParameters.size(); i++) {
				System.out.print(
						" -" + newAlgorithm.numericalParameters.get(i).parameter + " " + vals[pos++]);
			}
			for (int i = 0; i < newAlgorithm.integerParameters.size(); i++) {
				System.out.print(
						" -" + newAlgorithm.integerParameters.get(i).parameter + " " + vals[pos++]);
			}
			for (int i = 0; i < newAlgorithm.nominalParameters.size(); i++) {
				System.out.print(
						" -" + newAlgorithm.nominalParameters.get(i).parameter + " " + vals[pos++]);
			}
			for (int i = 0; i < newAlgorithm.booleanParameters.size(); i++) {
				if (vals[pos] == 1) {
					System.out.print(" -" + newAlgorithm.booleanParameters.get(i).parameter);
				}
				pos++;
			}
			System.out.println("\t => \t Silhouette: " + prediction);

			// TODO if ensemble empty, we could also just fill
			if (Double.isNaN(prediction)) {
				return;
			}

			// if we still have open slots in the ensemble (not full)
			if (this.ensemble.size() < this.settings.ensembleSize) {
				System.out.println("Ensemble not full. Add configuration as new algorithm.");

				// add to ensemble
				this.ensemble.add(newAlgorithm);

				// update current silhouettes with the prediction
				silhs.add(prediction);

			} else if (prediction > silh.getMinValue(0)) {
				// if the predicted performance is better than the one we have in the ensemble

				// proportionally sample a configuration that will be replaced
				int replaceIdx = EnsembleClustererAbstract.sampleProportionally(silhs);
				System.out.println(
						"Ensemble already full but new configuration is promising! Replace algorithm: " + replaceIdx);

				// replace in ensemble
				this.ensemble.set(replaceIdx, newAlgorithm);

				// update current silhouettes with the prediction
				silhs.set(replaceIdx, prediction);
			}
		}

	}

	// sample an index from a list of values, proportionally to the respective value
	static int sampleProportionally(ArrayList<Double> values) {
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
// System.out.println(ClassOption.stripPackagePrefix(this.ensemble[i].getClass().getName(),
// Clusterer.class)); // print class
// System.out.println(this.ensemble[i].getOptions().getAsCLIString()); // print
// non-default options
