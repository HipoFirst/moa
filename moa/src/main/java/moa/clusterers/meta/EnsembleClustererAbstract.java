package moa.clusterers.meta;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
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
import moa.clusterers.clustream.WithKmeans;
import moa.clusterers.clustree.ClusTree;
import moa.clusterers.denstream.WithDBSCAN;
import moa.clusterers.dstream.Dstream;
import moa.clusterers.kmeanspm.BICO;
import moa.clusterers.streamkm.StreamKM;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;
import moa.streams.clustering.ClusteringStream;
import moa.streams.clustering.RandomRBFGeneratorEvents;
import moa.streams.clustering.SimpleCSVStream;
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
	public boolean fixed;
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
	public boolean keepCurrentModel;
	public double lambda;
	public boolean preventAlgorithmDeath;
	public boolean reinitialiseWithClusters;
	public boolean evaluateMacro;
	public boolean keepGlobalIncumbent;
	public boolean keepAlgorithmIncumbents;
	public boolean keepInitialConfigurations;
	public boolean useTestEnsemble;
}

public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int iteration;
	int instancesSeen;
	int iter;
	public int bestModel;
	public ArrayList<Algorithm> ensemble;
	public ArrayList<Algorithm> candidateEnsemble;
	public ArrayList<DataPoint> windowPoints;
	HashMap<String, AdaptiveRandomForestRegressor> ARFregs = new HashMap<String, AdaptiveRandomForestRegressor>();
	GeneralConfiguration settings;
	ArrayList<Double> silhouettes;
	int verbose = 0;

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
			// this.ensemble.get(i).clusterer.resetLearning();
			this.ensemble.get(i).init();
		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		// it appears to use numAttributes as the index when no class exists
		if (inst.classIndex() < inst.numAttributes()) {
			inst.deleteAttributeAt(inst.classIndex()); // remove class label
		}

		DataPoint point = new DataPoint(inst, instancesSeen); // create data points from instance
		this.windowPoints.add(point); // remember points of the current window
		this.instancesSeen++;

		// train all models with the instance
		for (int i = 0; i < this.ensemble.size(); i++) {
			this.ensemble.get(i).clusterer.trainOnInstance(inst);
		}

		if (this.settings.useTestEnsemble && this.candidateEnsemble.size() > 0) {
			// train all models with the instance
			for (int i = 0; i < this.candidateEnsemble.size(); i++) {
				this.candidateEnsemble.get(i).clusterer.trainOnInstance(inst);
			}
		}

		// every windowSize we update the configurations
		if (this.instancesSeen % this.settings.windowSize == 0) {
			if (this.verbose >= 1) {
				System.out.println(" ");
				System.out.println("-------------- Processed " + instancesSeen + " Instances --------------");
			}

			updateConfiguration(); // update configuration
		}

	}

	protected void updateConfiguration() {
		// init evaluation measure (silhouette for now)
		if (this.verbose >= 2) {
			System.out.println(" ");
			System.out.println("---- Evaluate performance of current ensemble:");
		}
		evaluatePerformance();

		if (this.settings.useTestEnsemble) {
			promoteCandidatesIntoEnsemble();
		}

		if (this.verbose >= 1) {
			System.out.println("Clusterer " + this.bestModel + " ("
					+ this.ensemble.get(this.bestModel).clusterer.getCLICreationString(Clusterer.class)
					+ ") is the active clusterer with silhouette: " + this.silhouettes.get(this.bestModel));
		}

		generateNewConfigurations();

		this.windowPoints.clear(); // flush the current window
		this.iter++;
	}

	protected void evaluatePerformance() {

		HashMap<String, Double> bestPerformanceValMap = new HashMap<String, Double>();
		HashMap<String, Integer> bestPerformanceIdxMap = new HashMap<String, Integer>();
		HashMap<String, Integer> algorithmCount = new HashMap<String, Integer>();

		this.silhouettes = new ArrayList<Double>(this.ensemble.size());
		double bestPerformance = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < this.ensemble.size(); i++) {

			// predict performance just for evaluation
			predictPerformance(this.ensemble.get(i));

			double performance = computeSilhouette(this.ensemble.get(i));
			this.silhouettes.add(performance);
			if(performance > bestPerformance){
				this.bestModel = i;
				bestPerformance = performance;
			}

			if (this.verbose >= 1) {
				System.out.println(i + ") " + this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class)
						+ "\t => \t Silhouette: " + performance);
			}

			String algorithm = this.ensemble.get(i).algorithm;
			if (!bestPerformanceIdxMap.containsKey(algorithm) || performance > bestPerformanceValMap.get(algorithm)) {
				bestPerformanceValMap.put(algorithm, performance); // best silhouette per algorithm
				bestPerformanceIdxMap.put(algorithm, i); // index of best silhouette per algorithm
			}
			// number of instances per algorithm in ensemble
			algorithmCount.put(algorithm, algorithmCount.getOrDefault(algorithm, 0) + 1);

			trainRegressor(this.ensemble.get(i), performance);
		}

		updateRemovalFlags(bestPerformanceValMap, bestPerformanceIdxMap, algorithmCount);
	}

	protected double computeSilhouette(Algorithm algorithm) {

		SilhouetteCoefficient silhouette = new SilhouetteCoefficient();
		// compare micro-clusters
		Clustering result = null;
		if (!this.settings.evaluateMacro) {
			result = algorithm.clusterer.getMicroClusteringResult();
		}
		// compare macro-clusters
		if (this.settings.evaluateMacro || result == null) {
			// this is also the fallback for algorithms which dont export micro clusters
			// Note: This is not a fair comparison but otherwise we would have to discard
			// these algorithms entirely.
			if (this.verbose >= 2)
				System.out.println("Micro-Cluster not available for "
						+ algorithm.clusterer.getCLICreationString(Clusterer.class) + ". Try Macro-Clusters instead.");
			result = algorithm.clusterer.getClusteringResult();
		}

		double performance;
		if (result == null) {
			throw new RuntimeException("Neither micro- nor macro clusters available for "
					+ algorithm.clusterer.getCLICreationString(Clusterer.class));
		} else if (result.size() == 0 || result.size() == 1) {
			performance = -1.0; // discourage solutions with no or a single cluster
		} else {
			// evaluate clustering using silhouette width
			silhouette.evaluateClustering(result, null, windowPoints);
			performance = silhouette.getLastValue(0);
			// if ownDistance == otherDistance == 0 the Silhouette will return NaN
			if (Double.isNaN(performance)) {
				performance = -1.0;
			}
		}
		algorithm.silhouette = performance;

		return performance;
	}

	protected void promoteCandidatesIntoEnsemble() {

		for (int i = 0; i < this.candidateEnsemble.size(); i++) {

			Algorithm newAlgorithm = this.candidateEnsemble.get(i);

			// predict performance just for evaluation
			predictPerformance(newAlgorithm);

			// evaluate
			double performance = computeSilhouette(newAlgorithm);

			if (this.verbose >= 1) {
				System.out.println("Test " + i + ") " + newAlgorithm.clusterer.getCLICreationString(Clusterer.class)
						+ "\t => \t Silhouette: " + performance);
			}

			// replace if better than existing
			if (this.ensemble.size() < this.settings.ensembleSize) {
				if (this.verbose >= 1) {
					System.out.println("Promote " + newAlgorithm.clusterer.getCLICreationString(Clusterer.class)
							+ " from test ensemble to the ensemble as new configuration");
				}

				this.silhouettes.add(newAlgorithm.silhouette);

				this.ensemble.add(newAlgorithm);

			} else if (performance > EnsembleClustererAbstract.getWorstSolution(this.silhouettes)) {

				HashMap<Integer, Double> replace = getReplaceMap(this.silhouettes);

				if (replace.size() == 0) {
					return;
				}

				int replaceIdx = EnsembleClustererAbstract.sampleInvertProportionally(replace);

				if (this.verbose >= 1) {
					System.out.println("Promote " + newAlgorithm.clusterer.getCLICreationString(Clusterer.class)
							+ " from test ensemble to the ensemble by replacing " + replaceIdx);
				}

				// update silhouettes
				this.silhouettes.set(replaceIdx, newAlgorithm.silhouette);

				// replace in ensemble
				this.ensemble.set(replaceIdx, newAlgorithm);
			}

		}
	}

	protected void trainRegressor(Algorithm algortihm, double performance) {
		double[] params = algortihm.getParamVector(1);
		params[params.length - 1] = performance; // add performance as class
		Instance inst = new DenseInstance(1.0, params);

		// add header to dataset TODO: do we need an attribute for the class label?
		Instances dataset = new Instances(null, algortihm.attributes, 0);
		dataset.setClassIndex(dataset.numAttributes()); // set class index to our performance feature
		inst.setDataset(dataset);

		// train adaptive random forest regressor based on performance of model
		this.ARFregs.get(algortihm.algorithm).trainOnInstanceImpl(inst);
	}

	protected void updateRemovalFlags(HashMap<String, Double> bestPerformanceValMap,
			HashMap<String, Integer> bestPerformanceIdxMap, HashMap<String, Integer> algorithmCount) {

		// reset flags
		for (Algorithm algorithm : ensemble) {
			algorithm.preventRemoval = false;
		}

		// only keep best overall algorithm
		if (this.settings.keepGlobalIncumbent) {
			this.ensemble.get(this.bestModel).preventRemoval = true;
		}

		// keep best instance per algorithm
		if (this.settings.keepAlgorithmIncumbents) {
			for (int idx : bestPerformanceIdxMap.values()) {
				this.ensemble.get(idx).preventRemoval = true;
			}
		}

		// keep all default configurations
		if (this.settings.keepInitialConfigurations) {
			for (Algorithm algorithm : this.ensemble) {
				if (algorithm.isDefault) {
					algorithm.preventRemoval = true;
				}
			}
		}

		// keep at least one instance per algorithm
		if (this.settings.preventAlgorithmDeath) {
			for (Algorithm algorithm : this.ensemble) {
				if (algorithmCount.get(algorithm.algorithm) == 1) {
					algorithm.preventRemoval = true;
				}
			}
		}
	}

	// predict performance of new configuration
	protected void generateNewConfigurations() {

		// get performance values
		if (this.settings.useTestEnsemble) {
			candidateEnsemble.clear();
		}

		for (int z = 0; z < this.settings.newConfigurations; z++) {

			if (this.verbose == 2) {
				System.out.println(" ");
				System.out.println("---- Sample new configuration " + z + ":");
			}

			int parentIdx = sampleParent(this.silhouettes);
			Algorithm newAlgorithm = sampleNewConfiguration(this.silhouettes, parentIdx);

			if (this.settings.useTestEnsemble) {
				if (this.verbose >= 1) {
					System.out.println("Based on " + parentIdx + " add "
							+ newAlgorithm.clusterer.getCLICreationString(Clusterer.class) + " to test ensemble");
				}
				candidateEnsemble.add(newAlgorithm);
			} else {
				double prediction = predictPerformance(newAlgorithm);

				if (this.verbose >= 1) {
					System.out.println("Based on " + parentIdx + " predict: "
							+ newAlgorithm.clusterer.getCLICreationString(Clusterer.class) + "\t => \t Silhouette: "
							+ prediction);
				}

				// the random forest only works with at least two training samples
				if (Double.isNaN(prediction)) {
					return;
				}

				// if we still have open slots in the ensemble (not full)
				if (this.ensemble.size() < this.settings.ensembleSize) {
					if (this.verbose >= 1) {
						System.out.println("Add configuration as new algorithm.");
					}

					// add to ensemble
					this.ensemble.add(newAlgorithm);

					// update current silhouettes with the prediction
					this.silhouettes.add(prediction);

				} else if (prediction > EnsembleClustererAbstract.getWorstSolution(this.silhouettes)) {
					// if the predicted performance is better than the one we have in the ensemble
					HashMap<Integer, Double> replace = getReplaceMap(this.silhouettes);

					if (replace.size() == 0) {
						return;
					}

					// int replaceIdx = getWorstSolutionIdx(silhs);
					int replaceIdx = EnsembleClustererAbstract.sampleInvertProportionally(replace);

					if (this.verbose >= 1) {
						System.out.println("Replace algorithm: " + replaceIdx);
					}

					// update current silhouettes with the prediction
					this.silhouettes.set(replaceIdx, prediction);

					// replace in ensemble
					this.ensemble.set(replaceIdx, newAlgorithm);
				}
			}

		}

	}

	protected int sampleParent(ArrayList<Double> silhs) {
		// copy existing clusterer configuration
		HashMap<Integer, Double> parents = new HashMap<Integer, Double>();
		for (int i = 0; i < silhs.size(); i++) {
			parents.put(i, silhs.get(i));
		}
		int parentIdx = EnsembleClustererAbstract.sampleProportionally(parents);

		return parentIdx;
	}

	protected Algorithm sampleNewConfiguration(ArrayList<Double> silhs, int parentIdx) {

		if (this.verbose >= 2) {
			System.out.println("Selected Configuration " + parentIdx + " as parent: "
					+ this.ensemble.get(parentIdx).clusterer.getCLICreationString(Clusterer.class));
		}
		Algorithm newAlgorithm = new Algorithm(this.ensemble.get(parentIdx), this.settings.lambda,
				this.settings.keepCurrentModel, this.settings.reinitialiseWithClusters, this.verbose);

		return newAlgorithm;
	}

	protected double predictPerformance(Algorithm newAlgorithm) {
		// create a data point from new configuration
		double[] params = newAlgorithm.getParamVector(0);
		Instance newInst = new DenseInstance(1.0, params);
		Instances newDataset = new Instances(null, newAlgorithm.attributes, 0);
		newDataset.setClassIndex(newDataset.numAttributes());
		newInst.setDataset(newDataset);

		// predict the performance of the new configuration using the trained adaptive
		// random forest
		double prediction = this.ARFregs.get(newAlgorithm.algorithm).getVotesForInstance(newInst)[0];

		newAlgorithm.prediction = prediction; // remember prediction

		return prediction;
	}

	// get mapping of algorithms and their silhouette that could be removed
	HashMap<Integer, Double> getReplaceMap(ArrayList<Double> silhs) {
		HashMap<Integer, Double> replace = new HashMap<Integer, Double>();

		double worst = EnsembleClustererAbstract.getWorstSolution(silhs);

		// replace solutions that cannot get worse first
		if (worst <= -1.0) {
			for (int i = 0; i < this.ensemble.size(); i++) {
				if (silhs.get(i) <= -1.0 && !this.ensemble.get(i).preventRemoval) {
					replace.put(i, silhs.get(i));
				}
			}
		}

		if (replace.size() == 0) {
			for (int i = 0; i < this.ensemble.size(); i++) {
				if (!this.ensemble.get(i).preventRemoval) {
					replace.put(i, silhs.get(i));
				}
			}
		}

		return replace;
	}

	// get lowest value in arraylist
	static double getWorstSolution(ArrayList<Double> values) {

		double min = Double.POSITIVE_INFINITY;
		for (int i = 0; i < values.size(); i++) {
			if (values.get(i) < min) {
				min = values.get(i);
			}
		}
		return (min);
	}

	// get lowest value in arraylist
	static int getWorstSolutionIdx(ArrayList<Double> values) {

		double min = Double.POSITIVE_INFINITY;
		int idx = -1;
		for (int i = 0; i < values.size(); i++) {
			if (values.get(i) < min) {
				min = values.get(i);
				idx = i;
			}
		}
		return (idx);
	}

	// sample an index from a list of values, inverse proportionally to the
	// respective value
	static int sampleInvertProportionally(HashMap<Integer, Double> values) {

		HashMap<Integer, Double> vals = new HashMap<Integer, Double>(values.size());

		for (int i : values.keySet()) {
			vals.put(i, -1 * values.get(i));
		}

		return (EnsembleClustererAbstract.sampleProportionally(vals));
	}

	// sample an index from a list of values, proportionally to the respective value
	static int sampleProportionally(HashMap<Integer, Double> values) {

		// get min
		double minVal = Double.POSITIVE_INFINITY;
		for (Double value : values.values()) {
			if (value < minVal) {
				minVal = value;
			}
		}

		// to have a positive range we shift here
		double shift = Math.abs(minVal) - minVal;

		double completeWeight = 0.0;
		for (Double value : values.values()) {
			completeWeight += value + shift;
		}

		// sample random number within range of total weight
		double r = Math.random() * completeWeight;
		double countWeight = 0.0;

		for (int j : values.keySet()) {
			countWeight += values.get(j) + shift;
			if (countWeight >= r) {
				return j;
			}
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

			// create the ensemble
			this.ensemble = new ArrayList<Algorithm>(this.settings.ensembleSize);
			// copy and initialise the provided starting configurations in the ensemble
			for (int i = 0; i < this.settings.algorithms.length; i++) {
				this.ensemble.add(new Algorithm(this.settings.algorithms[i]));
			}

			if (this.settings.useTestEnsemble) {
				this.candidateEnsemble = new ArrayList<Algorithm>(this.settings.newConfigurations);
			}

			// create one regressor per algorithm
			for (int i = 0; i < this.settings.algorithms.length; i++) {
				AdaptiveRandomForestRegressor ARFreg = new AdaptiveRandomForestRegressor();
				ARFreg.prepareForUse();
				this.ARFregs.put(this.settings.algorithms[i].algorithm, ARFreg);
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		super.prepareForUseImpl(monitor, repository);

	}

	public static void main(String[] args) throws FileNotFoundException {

		ArrayList<ClusteringStream> streams = new ArrayList<ClusteringStream>();
		SimpleCSVStream file;

		RandomRBFGeneratorEvents rbf = new RandomRBFGeneratorEvents();
		rbf.modelRandomSeedOption.setValue(2410);
		rbf.eventFrequencyOption.setValue(30000);
		rbf.eventDeleteCreateOption.setValue(true);
		rbf.eventMergeSplitOption.setValue(true);
		streams.add(rbf);

		file = new SimpleCSVStream();
		file.csvFileOption = new FileOption("", 'z', "",
		"sensor_relevant_standardized.csv",
		"", false);
		streams.add(file);

		file = new SimpleCSVStream();
		file.csvFileOption = new FileOption("", 'z', "",
		"powersupply_relevant_standardized.csv", "", false);
		streams.add(file);

		file = new SimpleCSVStream();
		file.csvFileOption = new FileOption("", 'z', "",
		"covertype_relevant_standardized.csv", "", false);
		streams.add(file);

		int[] lengths = { 2000000, 2219803, 29928, 581012 };
		String[] names = { "RBF", "sensor", "powersupply", "covertype" };
		int[] dimensions = { 2, 4, 2, 10 };

		// int[] lengths = { 50000 };
		// String[] names = { "RBF" };
		// int[] dimensions = { 2 };

		int windowSize = 1000;

		for (int s = 0; s < streams.size(); s++) {

			ArrayList<AbstractClusterer> algorithms = new ArrayList<AbstractClusterer>();

			// run confstream algorithm
			ConfStream confstream = new ConfStream();
			confstream.fileOption.setValue("settings_confStream.json");
			algorithms.add(confstream);

			// compare to individual algorithms
			WithDBSCAN denstream = new WithDBSCAN();
			algorithms.add(denstream);

			ClusTree clustree = new ClusTree();
			algorithms.add(clustree);

			WithKmeans clustream = new WithKmeans();
			algorithms.add(clustream);

			BICO bico = new BICO();
			bico.numDimensionsOption.setValue(dimensions[s]);
			algorithms.add(bico);

			Dstream dstream = new Dstream(); // only macro
			algorithms.add(dstream);

			StreamKM streamkm = new StreamKM(); // only macro
			streamkm.lengthOption.setValue(lengths[s]);
			algorithms.add(streamkm);

			// confstream with predictor
			ConfStream confstreamusePredictor = new ConfStream();
			confstreamusePredictor.fileOption.setValue("settings_confStream_usePredictor.json");
			algorithms.add(confstreamusePredictor);

			// confstream without keeping the starting configuration
			ConfStream confstreamNoInitial = new ConfStream();
			confstreamNoInitial.fileOption.setValue("settings_confStream_noInitial.json");
			algorithms.add(confstreamNoInitial);

			// confstream without keeping the starting configuration or the algorithm incumbent or the overall incumbent
			ConfStream confstreamNoIncumbentAndAlgorithmIncumbentsAndInitial = new ConfStream();
			confstreamNoIncumbentAndAlgorithmIncumbentsAndInitial.fileOption.setValue("settings_confStream_noIncumbentAndAlgorithmIncumbentsAndInitial.json");
			algorithms.add(confstreamNoIncumbentAndAlgorithmIncumbentsAndInitial);

			// no algorithm incumbent, no default
			ConfStream confstreamNoAlgorithmIncumbentsAndDefault = new ConfStream();
			confstreamNoAlgorithmIncumbentsAndDefault.fileOption.setValue("settings_confStream_noAlgorithmIncumbentsAndInitial.json");
			algorithms.add(confstreamNoAlgorithmIncumbentsAndDefault);

			// run confstream only on single algorithms
			ConfStream confstreamDenstream = new ConfStream();
			confstreamDenstream.fileOption.setValue("settings_denstream.json");
			algorithms.add(confstreamDenstream);

			ConfStream confstreamClustree = new ConfStream();
			confstreamClustree.fileOption.setValue("settings_clustree.json");
			algorithms.add(confstreamClustree);

			ConfStream confstreamClustream = new ConfStream();
			confstreamClustream.fileOption.setValue("settings_clustream.json");
			algorithms.add(confstreamClustream);

			ConfStream confstreamBico = new ConfStream();
			confstreamBico.fileOption.setValue("settings_bico.json");
			algorithms.add(confstreamBico);

			// compare on-the-fly adaption to reinitialisation with micro to reset
			ConfStream confStreamReinit = new ConfStream();
			confStreamReinit.fileOption.setValue("settings_confStream_reinitialiseModel.json");
			algorithms.add(confStreamReinit);

			ConfStream confStreamReset = new ConfStream();
			confStreamReset.fileOption.setValue("settings_confStream_resetModel.json");
			algorithms.add(confStreamReset);

			ConfStream denStreamNoReinit = new ConfStream();
			denStreamNoReinit.fileOption.setValue("settings_denstream_reinitialiseModel.json");
			algorithms.add(denStreamNoReinit);

			ConfStream denStreamReinit = new ConfStream();
			denStreamReinit.fileOption.setValue("settings_denstream_resetModel.json");
			algorithms.add(denStreamReinit);

			// run algorithms with already optimised parameters
			if (names[s].equals("sensor") || names[s].equals("covertype")) {
				WithDBSCAN denstreamOptim = new WithDBSCAN();
				WithKmeans clustreamOptim = new WithKmeans();
				ClusTree clustreeOptim = new ClusTree();
				// Dstream dstreamOptim = new Dstream(); // only macro
				if (names[s].equals("sensor")) {
					denstreamOptim.epsilonOption.setValue(0.02);
					denstreamOptim.muOption.setValue(2.78);
					denstreamOptim.betaOption.setValue(0.69);
					clustreamOptim.kernelRadiFactorOption.setValue(7);
					clustreeOptim.maxHeightOption.setValue(9);
					// dstreamOptim.cmOption.setValue(1.38);
					// dstreamOptim.clOption.setValue(1.25);
				} else if (names[s].equals("covertype")) {
					denstreamOptim.epsilonOption.setValue(0.42);
					denstreamOptim.muOption.setValue(2.51);
					denstreamOptim.betaOption.setValue(0.33);
					clustreamOptim.kernelRadiFactorOption.setValue(3);
					clustreeOptim.maxHeightOption.setValue(6);
					// dstreamOptim.cmOption.setValue(1.65);
					// dstreamOptim.clOption.setValue(0.34);
				}
				algorithms.add(denstreamOptim);
				algorithms.add(clustreamOptim);
				algorithms.add(clustreeOptim);
				// algorithms.add(dstreamOptim);
			}

			System.out.println("Stream: " + names[s]);
			streams.get(s).prepareForUse();
			streams.get(s).restart();

			for (int a = 0; a < algorithms.size(); a++) {
				System.out.println("Algorithm: " + algorithms.get(a).getCLICreationString(Clusterer.class));

				algorithms.get(a).prepareForUse();

				// TODO these are super ugly special cases
				if (algorithms.get(a) instanceof EnsembleClustererAbstract) {
					EnsembleClustererAbstract confStream = (EnsembleClustererAbstract) algorithms.get(a);
					for (Algorithm alg : confStream.ensemble) {
						for (IParameter param : alg.parameters) {
							if (alg.clusterer instanceof StreamKM && param.getParameter().equals("l")) {
								IntegerParameter integerParam = (IntegerParameter) param;
								integerParam.setValue(lengths[s]);
							}
							if (alg.clusterer instanceof BICO && param.getParameter().equals("d")) {
								IntegerParameter integerParam = (IntegerParameter) param;
								integerParam.setValue(dimensions[s]);
							}
						}
					}
				}
				algorithms.get(a).resetLearningImpl();
				streams.get(s).restart();

				File resultFile = new File(names[s] + "_"
						+ algorithms.get(a).getCLICreationString(Clusterer.class)
						+ ".txt");
				PrintWriter resultWriter = new PrintWriter(resultFile);

				PrintWriter ensembleWriter = null;
				PrintWriter predictionWriter = null;

				// header of proportion file
				if (algorithms.get(a) instanceof EnsembleClustererAbstract) {

					EnsembleClustererAbstract confStream = (EnsembleClustererAbstract) algorithms.get(a);

					// init prediction for ensemble algorithms writer
					File ensembleFile = new File(
							names[s] + "_" + algorithms.get(a).getCLICreationString(Clusterer.class) + "_ensemble.txt");
					ensembleWriter = new PrintWriter(ensembleFile);

					ensembleWriter.println("points\tidx\tAlgorithm");

					for (int i = 0; i < confStream.ensemble.size(); i++) {
						ensembleWriter.print(0);
						ensembleWriter.print("\t" + i);
						ensembleWriter.println("\t" + confStream.ensemble.get(i).algorithm);
					}
					ensembleWriter.flush();

					// init writer for individual prediction comparison
					File predictionFile = new File(names[s] + "_"
							+ algorithms.get(a).getCLICreationString(Clusterer.class) + "_prediction.txt");
					predictionWriter = new PrintWriter(predictionFile);
					predictionWriter.println("points\tidx\talgorithm\tsilhouette\tprediction");
				}

				// header of result file
				resultWriter.println("points\tsilhouette");

				ArrayList<DataPoint> windowPoints = new ArrayList<DataPoint>(windowSize);
				// ArrayList<Instance> windowInstances = new ArrayList<Instance>(windowSize);
				for (int d = 1; d < lengths[s]; d++) {
					Instance inst = streams.get(s).nextInstance().getData();

					// apparently numAttributes is the class index when no class exists
					if (inst.classIndex() < inst.numAttributes()) {
						inst.deleteAttributeAt(inst.classIndex()); // remove class label
					}
					DataPoint point = new DataPoint(inst, d);
					windowPoints.add(point);
					// windowInstances.add(inst);
					algorithms.get(a).trainOnInstanceImpl(inst);

					// if (d % windowSize == 0 && d != windowSize) {
					if (d % windowSize == 0) {

						SilhouetteCoefficient silh = new SilhouetteCoefficient();

						Clustering result = null;
						boolean evaluateMacro = false;

						// compare micro-clusters
						if (!evaluateMacro) {
							result = algorithms.get(a).getMicroClusteringResult();
						}
						// compare macro-clusters
						if (evaluateMacro || result == null) {
							result = algorithms.get(a).getClusteringResult();
						}

						resultWriter.print(d);
						resultWriter.print("\t");

						if (result == null) {
							resultWriter.print(-1.0);
						} else {
							silh.evaluateClustering(result, null, windowPoints);

							if (result.size() == 0 || result.size() == 1) {
								resultWriter.print("nan");
							} else {
								resultWriter.printf("%f", silh.getLastValue(0));
							}
						}
						resultWriter.print("\n");

						if (algorithms.get(a) instanceof EnsembleClustererAbstract) {
							EnsembleClustererAbstract confStream = (EnsembleClustererAbstract) algorithms.get(a);
							Algorithm alg = confStream.ensemble.get(confStream.bestModel);

							File paramFile = new File(
									names[s] + "_" + algorithms.get(a).getCLICreationString(Clusterer.class) + "_"
											+ alg.algorithm + ".txt");

							PrintWriter paramWriter = new PrintWriter(new FileOutputStream(paramFile, true)); // append

							// add header to param file
							try {
								BufferedReader br = new BufferedReader(new FileReader(paramFile));
								if (br.readLine() == null) {
									paramWriter.print("points\tsilhouette");
									for (int p = 0; p < alg.parameters.length; p++) {
										paramWriter.print("\t" + alg.parameters[p].getParameter());
									}
									paramWriter.print("\n");
								}
								br.close();
							} catch (IOException e) {
							}

							// add param values
							paramWriter.print(d);
							paramWriter.printf("\t%f", silh.getLastValue(0));
							for (int p = 0; p < alg.parameters.length; p++) {
								paramWriter.print("\t" + alg.parameters[p].getValue());
							}
							paramWriter.print("\n");
							paramWriter.close();

							// ensemble compositions
							for (int i = 0; i < confStream.ensemble.size(); i++) {
								ensembleWriter.print(d);
								ensembleWriter.print("\t" + i);
								ensembleWriter.println("\t" + confStream.ensemble.get(i).algorithm);
							}
							ensembleWriter.flush();

							for (int i = 0; i < confStream.ensemble.size(); i++) {
								predictionWriter.print(d);
								predictionWriter.print("\t" + i);
								predictionWriter.print("\t"
										+ confStream.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class));
								predictionWriter.printf("\t%f", confStream.ensemble.get(i).silhouette);
								predictionWriter.printf("\t%f", +confStream.ensemble.get(i).prediction);
								predictionWriter.print("\n");
							}
							predictionWriter.flush();
						}

						// // then train
						// for(Instance inst2 : windowInstances){
						// algorithms.get(a).trainOnInstanceImpl(inst2);
						// }

						// windowInstances.clear();
						windowPoints.clear();
						resultWriter.flush();
					}

					if (d % 10000 == 0) {
						System.out.println("Observation: " + d);
					}
				}
				resultWriter.close();
				if (algorithms.get(a) instanceof EnsembleClustererAbstract) {
					ensembleWriter.close();
					predictionWriter.close();
				}
			}

		}
	}
}
