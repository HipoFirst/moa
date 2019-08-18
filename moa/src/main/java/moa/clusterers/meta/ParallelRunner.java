package moa.clusterers.meta;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;

import com.github.javacliparser.FileOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;
import moa.clusterers.clustream.WithKmeans;
import moa.clusterers.clustree.ClusTree;
import moa.clusterers.denstream.WithDBSCAN;
import moa.clusterers.dstream.Dstream;
import moa.clusterers.kmeanspm.BICO;
import moa.clusterers.streamkm.StreamKM;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;
import moa.streams.clustering.SimpleCSVStream;


public class ParallelRunner {

	public static void main(String[] args) throws FileNotFoundException, InterruptedException, ExecutionException {

		int numberOfCores = Integer.parseInt(args[0]);
		String name = args[1];
		String filename = args[2];
		int length = Integer.parseInt(args[3]);
		int dimensions = Integer.parseInt(args[4]);
	
		// int numberOfCores = 1;

		// String name =  "RBF";
		// String filename = "RBF_relevant.csv";
		// int length = 2000000;
		// int dimensions = 2;

		// String name =  "sensor";
		// String filename = "sensor_relevant_standardized.csv";
		// int length = 2219803;
		// int dimensions = 4;

		// String name =  "powersupply";
		// String filename = "powersupply_relevant_standardized.csv";
		// int length = 29928;
		// int dimensions = 2;

		// String name =  "covertype";
		// String filename = "covertype_relevant_standardized.csv";
		// int length = 581012;
		// int dimensions = 10;

		int windowSize = 1000;

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
		bico.numDimensionsOption.setValue(dimensions);
		algorithms.add(bico);

		// confstream with predictor
		ConfStream confstreamusePredictor = new ConfStream();
		confstreamusePredictor.fileOption.setValue("settings_confStream_usePredictor.json");
		algorithms.add(confstreamusePredictor);

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

		// run algorithms with already optimised parameters
		WithDBSCAN denstreamcRand = new WithDBSCAN();
		WithKmeans clustreamcRand = new WithKmeans();
		ClusTree clustreecRand = new ClusTree();

		if (name.equals("sensor")) {
			denstreamcRand.epsilonOption.setValue(0.02);
			denstreamcRand.muOption.setValue(2.78);
			denstreamcRand.betaOption.setValue(0.69);
			denstreamcRand.lambdaOption.setValue(0.001);
			clustreamcRand.kernelRadiFactorOption.setValue(7);
			clustreecRand.maxHeightOption.setValue(9);

			algorithms.add(denstreamcRand);
			algorithms.add(clustreamcRand);
			algorithms.add(clustreecRand);

		} else if (name.equals("covertype")) {
			denstreamcRand.epsilonOption.setValue(0.42);
			denstreamcRand.muOption.setValue(2.51);
			denstreamcRand.betaOption.setValue(0.33);
			denstreamcRand.lambdaOption.setValue(0.001);
			clustreamcRand.kernelRadiFactorOption.setValue(3);
			clustreecRand.maxHeightOption.setValue(6);

			algorithms.add(denstreamcRand);
			algorithms.add(clustreamcRand);
			algorithms.add(clustreecRand);
		}

		WithDBSCAN denstreamIrace = new WithDBSCAN();
		ClusTree clustreeIrace = new ClusTree();
		WithKmeans clustreamIrace = new WithKmeans();
		BICO bicoIrace = new BICO();

		if (name.equals("RBF")) {
			denstreamIrace.epsilonOption.setValue(0.0757); // e
			denstreamIrace.betaOption.setValue(0.3205); // b
			denstreamIrace.muOption.setValue(2913.1242); // m
			denstreamIrace.offlineOption.setValue(16.489); // o
			denstreamIrace.lambdaOption.setValue(0.1037); // l
			// clustreeIrace.maxHeightOption.setValue(); // H
			// clustreeIrace.breadthFirstStrategyOption.setValue(); // B
			clustreamIrace.kOption.setValue(5); // k
			clustreamIrace.maxNumKernelsOption.setValue(100); // m
			clustreamIrace.kernelRadiFactorOption.setValue(2); // t
			// bicoIrace.numClustersOption.setValue(); // k
			// bicoIrace.maxNumClusterFeaturesOption.setValue(); // n
			// bicoIrace.numProjectionsOption.setValue(); // p
			algorithms.add(denstreamIrace);
			algorithms.add(clustreamIrace);
		} else if (name.equals("sensor")) {
			denstreamIrace.epsilonOption.setValue(0.8014); // e
			denstreamIrace.betaOption.setValue(0.2593); // b
			denstreamIrace.muOption.setValue(9085.1493); // m
			denstreamIrace.offlineOption.setValue(7.0789); // o
			denstreamIrace.lambdaOption.setValue(0.0744); // l
			clustreeIrace.maxHeightOption.setValue(3); // H
			clustreeIrace.breadthFirstStrategyOption.setValue(true); // B
			clustreamIrace.kOption.setValue(8); // k
			clustreamIrace.maxNumKernelsOption.setValue(98); // m
			clustreamIrace.kernelRadiFactorOption.setValue(2); // t
			// bicoIrace.numClustersOption.setValue(); // k
			// bicoIrace.maxNumClusterFeaturesOption.setValue(); // n
			// bicoIrace.numProjectionsOption.setValue(); // p
			algorithms.add(denstreamIrace);
			algorithms.add(clustreamIrace);
		} else if (name.equals("powersupply")) {
			denstreamIrace.epsilonOption.setValue(0.3469); // e
			denstreamIrace.betaOption.setValue(0.0174); // b
			denstreamIrace.muOption.setValue(4027.0768); // m
			denstreamIrace.offlineOption.setValue(7.5439); // o
			denstreamIrace.lambdaOption.setValue(0.8842); // l
			clustreeIrace.maxHeightOption.setValue(1); // H
			clustreeIrace.breadthFirstStrategyOption.setValue(true); // B
			clustreamIrace.kOption.setValue(5); // k
			clustreamIrace.maxNumKernelsOption.setValue(200); // m
			clustreamIrace.kernelRadiFactorOption.setValue(2); // t
			bicoIrace.numClustersOption.setValue(14); // k
			bicoIrace.maxNumClusterFeaturesOption.setValue(53); // n
			bicoIrace.numProjectionsOption.setValue(3); // p
			algorithms.add(denstreamIrace);
			algorithms.add(clustreamIrace);
			algorithms.add(clustreeIrace);
			algorithms.add(bicoIrace);
		} else if (name.equals("covertype")) {
			denstreamIrace.epsilonOption.setValue(0.5493); // e
			denstreamIrace.betaOption.setValue(0.6114); // b
			denstreamIrace.muOption.setValue(282.3994); // m
			denstreamIrace.offlineOption.setValue(3.3658); // o
			denstreamIrace.lambdaOption.setValue(0.1069); // l
			clustreeIrace.maxHeightOption.setValue(1); // H
			clustreeIrace.breadthFirstStrategyOption.setValue(true); // B
			clustreamIrace.kOption.setValue(3); // k
			clustreamIrace.maxNumKernelsOption.setValue(4); // m
			clustreamIrace.kernelRadiFactorOption.setValue(2); // t
			bicoIrace.numClustersOption.setValue(16); // k
			bicoIrace.maxNumClusterFeaturesOption.setValue(637); // n
			bicoIrace.numProjectionsOption.setValue(2); // p
			algorithms.add(denstreamIrace);
			algorithms.add(clustreamIrace);
			algorithms.add(clustreeIrace);
			algorithms.add(bicoIrace);
		}

		// Dstream dstream = new Dstream(); // only macro
		// algorithms.add(dstream);

		// StreamKM streamkm = new StreamKM(); // only macro
		// streamkm.lengthOption.setValue(length);
		// algorithms.add(streamkm);

		// // confstream without keeping the starting configuration
		// ConfStream confstreamNoInitial = new ConfStream();
		// confstreamNoInitial.fileOption.setValue("settings_confStream_noInitial.json");
		// algorithms.add(confstreamNoInitial);

		// // confstream without keeping the starting configuration or the algorithm
		// // incumbent or the overall incumbent
		// ConfStream confstreamNoIncumbentAndAlgorithmIncumbentsAndInitial = new
		// ConfStream();
		// confstreamNoIncumbentAndAlgorithmIncumbentsAndInitial.fileOption
		// .setValue("settings_confStream_noIncumbentAndAlgorithmIncumbentsAndInitial.json");
		// algorithms.add(confstreamNoIncumbentAndAlgorithmIncumbentsAndInitial);

		// // no algorithm incumbent, no default
		// ConfStream confstreamNoAlgorithmIncumbentsAndDefault = new ConfStream();
		// confstreamNoAlgorithmIncumbentsAndDefault.fileOption
		// .setValue("settings_confStream_noAlgorithmIncumbentsAndInitial.json");
		// algorithms.add(confstreamNoAlgorithmIncumbentsAndDefault);

		// // compare on-the-fly adaption to reinitialisation with micro to reset
		// ConfStream confStreamReinit = new ConfStream();
		// confStreamReinit.fileOption.setValue("settings_confStream_reinitialiseModel.json");
		// algorithms.add(confStreamReinit);

		// ConfStream confStreamReset = new ConfStream();
		// confStreamReset.fileOption.setValue("settings_confStream_resetModel.json");
		// algorithms.add(confStreamReset);

		// ConfStream denStreamNoReinit = new ConfStream();
		// denStreamNoReinit.fileOption.setValue("settings_denstream_reinitialiseModel.json");
		// algorithms.add(denStreamNoReinit);

		// ConfStream denStreamReinit = new ConfStream();
		// denStreamReinit.fileOption.setValue("settings_denstream_resetModel.json");
		// algorithms.add(denStreamReinit);





		ForkJoinPool myPool = new ForkJoinPool(numberOfCores);
		myPool.submit(() -> algorithms.parallelStream().forEach((algorithm) -> {

			try {

				SimpleCSVStream stream = new SimpleCSVStream();
				stream.csvFileOption = new FileOption("", 'z', "", filename, "", false);

				System.out.println("Starting Stream: " + name);
				stream.prepareForUse();
				stream.restart();

				System.out.println("Starting Algorithm: " + algorithm.getCLICreationString(Clusterer.class));

				algorithm.prepareForUse();

				// TODO these are super ugly special cases
				if (algorithm instanceof EnsembleClustererAbstract) {
					EnsembleClustererAbstract confStream = (EnsembleClustererAbstract) algorithm;
					for (Algorithm alg : confStream.ensemble) {
						for (IParameter param : alg.parameters) {
							if (alg.clusterer instanceof StreamKM && param.getParameter().equals("l")) {
								IntegerParameter integerParam = (IntegerParameter) param;
								integerParam.setValue(length);
							}
							if (alg.clusterer instanceof BICO && param.getParameter().equals("d")) {
								IntegerParameter integerParam = (IntegerParameter) param;
								integerParam.setValue(dimensions);
							}
						}
					}
				}
				algorithm.resetLearningImpl();
				stream.restart();

				File resultFile = new File(name + "_" + algorithm.getCLICreationString(Clusterer.class) + ".txt");
				PrintWriter resultWriter = new PrintWriter(resultFile);

				PrintWriter ensembleWriter = null;
				PrintWriter predictionWriter = null;

				// header of proportion file
				if (algorithm instanceof EnsembleClustererAbstract) {

					EnsembleClustererAbstract confStream = (EnsembleClustererAbstract) algorithm;

					// init prediction for ensemble algorithms writer
					File ensembleFile = new File(
							name + "_" + algorithm.getCLICreationString(Clusterer.class) + "_ensemble.txt");
					ensembleWriter = new PrintWriter(ensembleFile);

					ensembleWriter.println("points\tidx\tAlgorithm");

					for (int i = 0; i < confStream.ensemble.size(); i++) {
						ensembleWriter.print(0);
						ensembleWriter.print("\t" + i);
						ensembleWriter.println("\t" + confStream.ensemble.get(i).algorithm);
					}
					ensembleWriter.flush();

					// init writer for individual prediction comparison
					File predictionFile = new File(
							name + "_" + algorithm.getCLICreationString(Clusterer.class) + "_prediction.txt");
					predictionWriter = new PrintWriter(predictionFile);
					predictionWriter.println("points\tidx\talgorithm\tsilhouette\tprediction");
				}

				// header of result file
				resultWriter.println("points\tsilhouette");

				ArrayList<DataPoint> windowPoints = new ArrayList<DataPoint>(windowSize);
				// ArrayList<Instance> windowInstances = new ArrayList<Instance>(windowSize);
				for (int d = 1; d < length; d++) {
					Instance inst = stream.nextInstance().getData();

					// apparently numAttributes is the class index when no class exists
					if (inst.classIndex() < inst.numAttributes()) {
						inst.deleteAttributeAt(inst.classIndex()); // remove class label
					}
					DataPoint point = new DataPoint(inst, d);
					windowPoints.add(point);
					// windowInstances.add(inst);
					algorithm.trainOnInstanceImpl(inst);

					// if (d % windowSize == 0 && d != windowSize) {
					if (d % windowSize == 0) {

						SilhouetteCoefficient silh = new SilhouetteCoefficient();

						Clustering result = null;
						boolean evaluateMacro = false;

						// compare micro-clusters
						if (!evaluateMacro) {
							result = algorithm.getMicroClusteringResult();
						}
						// compare macro-clusters
						if (evaluateMacro || result == null) {
							result = algorithm.getClusteringResult();
						}

						resultWriter.print(d);
						resultWriter.print("\t");

						if (result == null) {
							resultWriter.print("nan");
						} else {
							silh.evaluateClustering(result, null, windowPoints);

							if (result.size() == 0 || result.size() == 1) {
								resultWriter.print("nan");
							} else {
								resultWriter.printf("%f", silh.getLastValue(0));
							}
						}
						resultWriter.print("\n");

						if (algorithm instanceof EnsembleClustererAbstract) {
							EnsembleClustererAbstract confStream = (EnsembleClustererAbstract) algorithm;
							Algorithm alg = confStream.ensemble.get(confStream.bestModel);

							File paramFile = new File(name + "_" + algorithm.getCLICreationString(Clusterer.class) + "_"
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
						// algorithm.trainOnInstanceImpl(inst2);
						// }

						// windowInstances.clear();
						windowPoints.clear();
						resultWriter.flush();
					}

					if (d % 10000 == 0) {
						System.out.println("Observation: " + d + " - Algorithm: " + algorithm.getCLICreationString(Clusterer.class));
					}
				}
				resultWriter.close();
				if (algorithm instanceof EnsembleClustererAbstract) {
					ensembleWriter.close();
					predictionWriter.close();
				}
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		})).get();
	}
}
