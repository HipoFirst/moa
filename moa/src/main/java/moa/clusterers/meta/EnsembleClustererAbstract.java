package moa.clusterers.meta;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

import com.github.javacliparser.FileOption;
import com.google.gson.Gson;
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
import moa.streams.clustering.RandomRBFGeneratorEvents;
import moa.tasks.TaskMonitor;

// The main flow is as follow:
// A json is read which contains the main settings and starting configurations / algorithms
// The json is used to initialize the three configuration classes below (same structure as json)
// From the json, we create the Algorithm and Parameter classes (depending on the type of the parameter) which form the ensemble of clusterers
// These classes are then used to cluster and evaluate the configurations
// When a new configuration is required, a parameter configuration is copied and the parameters manipulated

// these classes are initialised by gson and contain the starting configurations
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

public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int iteration;
	int instancesSeen;
	int iter;
	int currentEnsembleSize;
	int bestModel;
	ArrayList<Algorithm> ensemble;
	ArrayList<DataPoint> windowPoints;
	HashMap<String, AdaptiveRandomForestRegressor> ARFregs = new HashMap<String, AdaptiveRandomForestRegressor>();
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
		this.iter = 0;
		this.windowPoints = new ArrayList<DataPoint>(this.settings.windowSize);

		// reset ARFrefs
		for (AdaptiveRandomForestRegressor ARFreg : this.ARFregs.values()) {
			ARFreg.resetLearning();
		}

		// reset individual clusterers
		for (int i = 0; i < this.ensemble.size(); i++) {
			this.ensemble.get(i).clusterer.resetLearning();
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if (inst.classIndex() < inst.numAttributes()) { // it appears to use numAttributes as the index when no class
														// exists
			inst.deleteAttributeAt(inst.classIndex()); // remove class label
		}
		DataPoint point = new DataPoint(inst, instancesSeen); // create data points from instance
		this.windowPoints.add(point); // remember points of the current window
		this.instancesSeen++;

		// train all models with the instance
		for (int i = 0; i < this.ensemble.size(); i++) {
			this.ensemble.get(i).clusterer.trainOnInstance(inst);
		}

		// every windowSize we update the configurations
		if (this.instancesSeen % this.settings.windowSize == 0) {
			System.out.println(" ");
			System.out.println(" ");
			System.out.println("-------------- Processed " + instancesSeen + " Instances --------------");
			updateConfiguration(); // update configuration
		}
	}

	protected void updateConfiguration() {
		// init evaluation measure (silhouette for now)
		SilhouetteCoefficient silh = new SilhouetteCoefficient();
		// train the random forest regressor based on the configuration performance
		// and find the best performing algorithm
		System.out.println(" ");
		System.out.println("---- Evaluate performance of current ensemble:");
		evaluatePerformance(silh);

		System.out.println("Clusterer " + this.bestModel + " is the active clusterer");

		// generate a new configuration and predict its performance using the random
		// forest regressor
		predictConfiguration(silh);

		this.windowPoints.clear(); // flush the current window
		this.iter++;
	}

	protected void evaluatePerformance(SilhouetteCoefficient silh) {

		double maxVal = -1 * Double.MAX_VALUE;
		for (int i = 0; i < this.ensemble.size(); i++) {
			// get micro clusters of this clusterer
			Clustering result = this.ensemble.get(i).clusterer.getMicroClusteringResult();
			if (result == null) {
				throw new RuntimeException("Micro clusters not available for "
						+ this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class));
			}

			// evaluate clustering using silhouette width
			silh.evaluateClustering(result, null, windowPoints);
			double performance = silh.getLastValue(0);
			System.out.println(i + ") " + this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class)
					+ "\t => \t Silhouette: " + performance);

			// find best clustering result among all algorithms
			if (performance > maxVal) {
				maxVal = performance;
				this.bestModel = i; // the clusterer with the best result becomes the active one
			}

			double[] params = this.ensemble.get(i).getParamVector(1);

			params[params.length - 1] = performance; // add performance as class
			Instance inst = new DenseInstance(1.0, params);

			// add header to dataset TODO: do we need an attribute for the class label?
			Instances dataset = new Instances(null, this.ensemble.get(i).attributes, 0);
			dataset.setClassIndex(dataset.numAttributes()); // set class index to our performance feature
			inst.setDataset(dataset);

			// train adaptive random forest regressor based on performance of model
			this.ARFregs.get(this.ensemble.get(i).algorithm).trainOnInstanceImpl(inst);
		}
	}

	// predict performance of new configuration
	protected void predictConfiguration(SilhouetteCoefficient silh) {

		// sample a parent configuration proportionally to its performance from the
		// ensemble
		ArrayList<Double> silhs = silh.getAllValues(0);

		for (int z = 0; z < this.settings.newConfigurations; z++) {

			System.out.println(" ");
			System.out.println("---- Sample new configuration " + z + ":");

			// copy existing clusterer configuration
			int parentIdx = EnsembleClustererAbstract.sampleProportionally(silhs);
			System.out.println("Selected Configuration " + parentIdx + " as parent: "
					+ this.ensemble.get(parentIdx).clusterer.getCLICreationString(Clusterer.class));
			Algorithm newAlgorithm = new Algorithm(this.ensemble.get(parentIdx));

			// sample new configuration from the parent
			newAlgorithm.sampleNewConfig(this.iter, this.settings.newConfigurations);
			newAlgorithm.init();

			double[] params = newAlgorithm.getParamVector(0);
			Instance newInst = new DenseInstance(1.0, params);
			Instances newDataset = new Instances(null, newAlgorithm.attributes, 0);
			newDataset.setClassIndex(newDataset.numAttributes());
			newInst.setDataset(newDataset);

			System.out.println(" ");
			System.out.println("---- Predict performance of new configuration:");
			double prediction = this.ARFregs.get(newAlgorithm.algorithm).getVotesForInstance(newInst)[0];
			System.out.println("Predict: " + newAlgorithm.clusterer.getCLICreationString(Clusterer.class)
					+ "\t => \t Silhouette: " + prediction);

			// random forest only works with at least two training samples
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

			this.instancesSeen = 0;
			this.bestModel = 0;
			this.iter = 0;
			this.windowPoints = new ArrayList<DataPoint>(this.settings.windowSize);

			// also create the ensemble which can be larger than the provided (starting)
			// configurations
			this.ensemble = new ArrayList<Algorithm>(this.settings.ensembleSize);
			// copy and initialise the provided starting configurations in the ensemble
			for (int i = 0; i < this.settings.algorithms.length; i++) {
				this.ensemble.add(new Algorithm(this.settings.algorithms[i]));
			}

			// create or reset one regressor per algorithm
			for (int i = 0; i < this.settings.algorithms.length; i++) {
				AdaptiveRandomForestRegressor ARFreg = new AdaptiveRandomForestRegressor();
				ARFreg.prepareForUse();
				this.ARFregs.put(this.settings.algorithms[i].algorithm, ARFreg);
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

		System.out.println("-------------");

		EnsembleClusterer algorithm2 = new EnsembleClusterer();
		RandomRBFGeneratorEvents stream2 = new RandomRBFGeneratorEvents();
		stream2.prepareForUse();
		algorithm2.prepareForUse();
		for (int i = 0; i < 1000000; i++) {
			Instance inst = stream2.nextInstance().getData();
			algorithm2.trainOnInstanceImpl(inst);
		}
		algorithm2.getClusteringResult();

	}

}
// System.out.println(ClassOption.stripPackagePrefix(this.ensemble[i].getClass().getName(),
// Clusterer.class)); // print class
// System.out.println(this.ensemble[i].getOptions().getAsCLIString()); // print
// non-default options
