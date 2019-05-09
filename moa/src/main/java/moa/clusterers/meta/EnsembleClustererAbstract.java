package moa.clusterers.meta;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import com.github.javacliparser.ClassOption;
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
import moa.clusterers.denstream.WithDBSCAN;
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
}

public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int iteration;
	int instancesSeen;
	int iter;
	int currentEnsembleSize;
	public int bestModel;
	public ArrayList<Algorithm> ensemble;
	public ArrayList<DataPoint> windowPoints;
	HashMap<String, AdaptiveRandomForestRegressor> ARFregs = new HashMap<String, AdaptiveRandomForestRegressor>();
	GeneralConfiguration settings;
	SilhouetteCoefficient silhouette;
	boolean verbose=false;

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
			this.ensemble.get(i).clusterer.resetLearning();
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

		// every windowSize we update the configurations
		if (this.instancesSeen % this.settings.windowSize == 0) {
			// System.out.println(" ");
			if(this.verbose) System.out.println(" ");
			if(this.verbose) System.out.println("-------------- Processed " + instancesSeen + " Instances --------------");

			updateConfiguration(); // update configuration
		}

	}

	protected void updateConfiguration() {
		// init evaluation measure (silhouette for now)
		this.silhouette = new SilhouetteCoefficient();
		// train the random forest regressor based on the configuration performance
		// and find the best performing algorithm
		// System.out.println(" ");
		// System.out.println("---- Evaluate performance of current ensemble:");
		evaluatePerformance();

		if(this.verbose) System.out.println("Clusterer " + this.bestModel + " is the active clusterer");
		// System.out.println(" ");

		// generate a new configuration and predict its performance using the random
		// forest regressor
		predictConfiguration();

		this.windowPoints.clear(); // flush the current window
		this.iter++;
	}

	protected void evaluatePerformance() {

		double maxVal = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < this.ensemble.size(); i++) {
			// get micro clusters of this clusterer
			Clustering result = this.ensemble.get(i).clusterer.getMicroClusteringResult();
			double performance;
			if (result == null) {
				throw new RuntimeException("Micro clusters not available for "
						+ this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class));
			} else if (result.size() == 0 || result.size() == 1) {
				performance = -1.0; // discourage solutions with no or a single cluster
				this.silhouette.addValue(0, performance);
			} else {
				// evaluate clustering using silhouette width
				this.silhouette.evaluateClustering(result, null, windowPoints);
				performance = this.silhouette.getLastValue(0);
				// if ownDistance == otherDistance == 0 the Silhouette will return NaN
				// TODO this is terribly inefficient just to replace an NaN value
				if (Double.isNaN(performance)) {
					ArrayList<Double> silhs = this.silhouette.getAllValues(0);
					this.silhouette = new SilhouetteCoefficient();
					for (double sil : silhs) {
						if (!Double.isNaN(sil)) {
							this.silhouette.addValue(0, sil);
						} else {
							this.silhouette.addValue(0, -1.0);
						}
					}
					continue;
				}
			}

			if(this.verbose) System.out.println(i + ") " + this.ensemble.get(i).clusterer.getCLICreationString(Clusterer.class)	+ "\t => \t Silhouette: " + performance);

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
	protected void predictConfiguration() {

		// sample a parent configuration proportionally to its performance from the
		// ensemble
		ArrayList<Double> silhs = this.silhouette.getAllValues(0);

		for (int z = 0; z < this.settings.newConfigurations; z++) {

			// System.out.println(" ");
			// System.out.println("---- Sample new configuration " + z + ":");

			// copy existing clusterer configuration
			int parentIdx = EnsembleClustererAbstract.sampleProportionally(silhs);
			// System.out.println("Selected Configuration " + parentIdx + " as parent: "
			// +
			// this.ensemble.get(parentIdx).clusterer.getCLICreationString(Clusterer.class));
			Algorithm newAlgorithm = new Algorithm(this.ensemble.get(parentIdx), this.settings.keepCurrentModel);

			// sample new configuration from the parent
			newAlgorithm.sampleNewConfig(this.iter, this.settings.newConfigurations, this.settings.keepCurrentModel);

			// create a data point from new configuration
			double[] params = newAlgorithm.getParamVector(0);
			Instance newInst = new DenseInstance(1.0, params);
			Instances newDataset = new Instances(null, newAlgorithm.attributes, 0);
			newDataset.setClassIndex(newDataset.numAttributes());
			newInst.setDataset(newDataset);

			// predict the performance of the new configuration using the trained adaptive random forest
			double prediction = this.ARFregs.get(newAlgorithm.algorithm).getVotesForInstance(newInst)[0];
			if(this.verbose) System.out.println("Based on " + parentIdx + " predict: " + newAlgorithm.clusterer.getCLICreationString(Clusterer.class) + "\t => \t Silhouette: " + prediction);

			// random forest only works with at least two training samples
			if (Double.isNaN(prediction)) {
				return;
			}

			// if we still have open slots in the ensemble (not full)
			if (this.ensemble.size() < this.settings.ensembleSize) {
				if(this.verbose) System.out.println("Add configuration as new algorithm.");

				// add to ensemble
				this.ensemble.add(newAlgorithm);

				// update current silhouettes with the prediction
				// silhs.add(prediction);

			} else if (prediction > EnsembleClustererAbstract.getWorstSolution(silhs)) { // || EnsembleClustererAbstract.getWorstSolution(silhs) == -1.0)
				// if the predicted performance is better than the one we have in the ensemble

				// proportionally sample a configuration that will be replaced
				int replaceIdx = EnsembleClustererAbstract.sampleInvertProportionally(silhs);

				// alternatively: get the worst result and replace it
				// int replaceIdx = EnsembleClustererAbstract.getWorstSolutionIdx(silhs);

				// update current silhouettes with the prediction
				// silhs.set(replaceIdx, prediction);

				if(this.verbose) System.out.println("Replace algorithm: " + replaceIdx);

				// replace in ensemble
				this.ensemble.set(replaceIdx, newAlgorithm);
			} //else {
				// Otherwise we discard the solution
				// System.out.println("Discard configuration.");
			//}
		}

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

	// get index of lowest value in arraylist
	static int getWorstSolutionIdx(ArrayList<Double> values) {

		double min = Double.POSITIVE_INFINITY;
		int minIdx = -1;
		for (int i = 0; i < values.size(); i++) {
			if (values.get(i) < min) {
				min = values.get(i);
				minIdx = i;
			}
		}
		return (minIdx);
	}


	// sample an index from a list of values, inverse proportionally to the
	// respective value
	static int sampleInvertProportionally(ArrayList<Double> values) {

		ArrayList<Double> vals = new ArrayList<Double>(values.size());

		for (int i = 0; i < values.size(); i++) {
			vals.add(-1 * values.get(i));
			// System.out.println(vals.get(i));
		}

		return (EnsembleClustererAbstract.sampleProportionally(vals));

	}

	// sample an index from a list of values, proportionally to the respective value
	static int sampleProportionally(ArrayList<Double> values) {

		// get min
		double minVal = Double.POSITIVE_INFINITY;
		for (Double value : values) {
			if (value < minVal) {
				minVal = value;
			}
		}
		minVal = Math.abs(minVal);

		// sum weights (shifted by abs(min) to have positive range)
		double completeWeight = 0.0;
		for (Double value : values) {
			completeWeight += value + minVal;
		}

		// sample random number within range of total weight
		double r = Math.random() * completeWeight;
		double countWeight = 0.0;
		for (int j = 0; j < values.size(); j++) {
			countWeight += values.get(j) + minVal;
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
		streams.add(new RandomRBFGeneratorEvents());
		streams.add(new SimpleCSVStream());
		((SimpleCSVStream) streams.get(streams.size() - 1)).csvFileOption = new FileOption("", 'z', "", "powersupply.csv", "", false);
		streams.add(new SimpleCSVStream());
		((SimpleCSVStream) streams.get(streams.size() - 1)).csvFileOption = new FileOption("", 'z', "", "sensor.csv", "", false);
		streams.add(new SimpleCSVStream());
		((SimpleCSVStream) streams.get(streams.size() - 1)).csvFileOption = new FileOption("", 'z', "", "covertype.csv", "", false);

		int[] lengths = { 2000000, 29928, 2219803, 581012 };
		String[] names = { "RBF", "powersupply", "sensor", "covertype" };
		
		int windowSize = 1000;

		ArrayList<AbstractClusterer> algorithms = new ArrayList<AbstractClusterer>();
		algorithms.add(new EnsembleClustererBlast());
		// algorithms.add(new EnsembleClustererMerge());
		algorithms.add(new WithDBSCAN());
		// algorithms.add(new WithKmeans());

		for (int s = 0; s < streams.size(); s++) {
			System.out.println("Stream: " + names[s]);
			streams.get(s).prepareForUse();
			streams.get(s).restart();

			for (int a = 0; a < algorithms.size(); a++) {
				System.out.println("Algorithm: " + ClassOption.stripPackagePrefix(algorithms.get(a).getClass().getName(), Clusterer.class));

				algorithms.get(a).prepareForUse();
				algorithms.get(a).resetLearningImpl();
				streams.get(s).restart();
				File f = new File(names[s] + "_"
						+ ClassOption.stripPackagePrefix(algorithms.get(a).getClass().getName(), Clusterer.class) + ".txt");
				PrintWriter pw = new PrintWriter(f);

				if (algorithms.get(a) instanceof EnsembleClustererAbstract) {
					pw.print("points\tsilhouette");
					EnsembleClustererAbstract alg = (EnsembleClustererAbstract) algorithms.get(a);
					Algorithm algorithm = alg.ensemble.get(alg.bestModel);
					for (int j = 0; j < algorithm.parameters.length; j++) {
						pw.print("\t" + algorithm.parameters[j].getParameter());
					}
				} else {
					pw.print("points\tsilhouette");
				}
				pw.print("\n");

				ArrayList<DataPoint> windowPoints = new ArrayList<DataPoint>(windowSize);
				for (int d = 1; d < lengths[s]; d++) {
					Instance inst = streams.get(s).nextInstance().getData();
					// apparently numAttributes is the class index when no class exists
					if (inst.classIndex() < inst.numAttributes()) {
						inst.deleteAttributeAt(inst.classIndex()); // remove class label
					}
					DataPoint point = new DataPoint(inst, d);
					windowPoints.add(point);
					algorithms.get(a).trainOnInstanceImpl(inst);
					if (d % windowSize == 0) {

						SilhouetteCoefficient silh = new SilhouetteCoefficient();
						Clustering result = algorithms.get(a).getMicroClusteringResult();
						silh.evaluateClustering(result, null, windowPoints);

						// System.out.println(silh.getLastValue(0));
						pw.print(d);
						pw.print("\t");

						if (result.size() == 0 || result.size() == 1) {
							pw.print(-1);
						} else{
							pw.print(silh.getLastValue(0));
						}

						if (algorithms.get(a) instanceof EnsembleClustererAbstract) {
							EnsembleClustererAbstract alg = (EnsembleClustererAbstract) algorithms.get(a);
							Algorithm algorithm = alg.ensemble.get(alg.bestModel);
							for (int p = 0; p < algorithm.parameters.length; p++) {
								pw.print("\t" + algorithm.parameters[p].getValue());
							}
						}
						pw.print("\n");

						windowPoints.clear();
						pw.flush();
					}

					if (d % 10000 == 0) {
						System.out.println("Observation: " + d);
					}
				}
				pw.close();
			}
		}
	}
}
