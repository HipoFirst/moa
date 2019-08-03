package moa.clusterers.meta;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;

import com.github.javacliparser.FileOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.clustream.WithKmeans;
import moa.clusterers.clustree.ClusTree;
import moa.clusterers.denstream.WithDBSCAN;
import moa.clusterers.dstream.Dstream;
import moa.clusterers.kmeanspm.BICO;
import moa.clusterers.streamkm.StreamKM;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;
import moa.streams.clustering.ClusteringStream;
import moa.streams.clustering.RandomRBFGeneratorEvents;
import moa.streams.clustering.SimpleCSVStream;

public abstract class Iracer {

	public static void main(String[] args) throws FileNotFoundException {

		int windowSize = 1000;

		String filename = args[0];
		String streamName = args[1];
		String algorithmName = args[2];

		ClusteringStream stream = null;
		AbstractClusterer algorithm = null;
		int length = 0;
		int dimensions =0;

		if(streamName.equals("rbf")){
			RandomRBFGeneratorEvents rbf = new RandomRBFGeneratorEvents();
			rbf.modelRandomSeedOption.setValue(2410);
			rbf.eventFrequencyOption.setValue(30000);
			rbf.eventDeleteCreateOption.setValue(true);
			rbf.eventMergeSplitOption.setValue(true);
			length = 2000000;
			dimensions = 2;
			stream = rbf;
		} else if(streamName.equals("sensor")){
			SimpleCSVStream file = new SimpleCSVStream();
			file.csvFileOption = new FileOption("", 'z', "", "sensor_relevant_standardized.csv", "", false);
			length = 2219803;
			dimensions = 4;
			stream = file;
		} else if(streamName.equals("powersupply")){
			SimpleCSVStream file = new SimpleCSVStream();
			file.csvFileOption = new FileOption("", 'z', "", "powersupply_relevant_standardized.csv", "", false);
			length = 29928;
			dimensions = 2;
			stream = file;
		} else if(streamName.equals("covertype")){
			SimpleCSVStream file = new SimpleCSVStream();
			file.csvFileOption = new FileOption("", 'z', "", "covertype_relevant_standardized.csv", "", false);
			length = 581012;
			dimensions = 10;
			stream = file;
		} else{
			throw new RuntimeException("Stream not found.");
		}

		if(algorithmName.equals("DenStream")){
			WithDBSCAN denstream = new WithDBSCAN();
			float e = Float.parseFloat(args[3]);
			float b = Float.parseFloat(args[4]);
			float m = Float.parseFloat(args[5]);
			float o = Float.parseFloat(args[6]);
			float l = Float.parseFloat(args[7]);
			denstream.epsilonOption.setValue(e);
			denstream.betaOption.setValue(b);
			denstream.muOption.setValue(m);
			denstream.offlineOption.setValue(o);
			denstream.lambdaOption.setValue(l);
			algorithm=denstream;
		} else if(algorithmName.equals("ClusTree")){
			ClusTree clustree = new ClusTree();
			int H = Integer.parseInt(args[3]);
			boolean B = Boolean.parseBoolean(args[4]);
			clustree.maxHeightOption.setValue(H); //TODO B option
			clustree.breadthFirstStrategyOption.setValue(B); //TODO B option
			algorithm=clustree;
		} else if(algorithmName.equals("CluStream")){
			WithKmeans clustream = new WithKmeans();
			int k = Integer.parseInt(args[3]);
			int m = Integer.parseInt(args[4]);
			int t = Integer.parseInt(args[5]);
			clustream.kOption.setValue(k);
			clustream.kernelRadiFactorOption.setValue(t);
			clustream.maxNumKernelsOption.setValue(m);
			algorithm=clustream;
		} else if(algorithmName.equals("BICO")){
			BICO bico = new BICO();
			int k = Integer.parseInt(args[3]);
			int n = Integer.parseInt(args[4]);
			int p = Integer.parseInt(args[5]);
			bico.numClustersOption.setValue(k);
			bico.maxNumClusterFeaturesOption.setValue(n);
			bico.numProjectionsOption.setValue(p);
			bico.numDimensionsOption.setValue(dimensions);
			algorithm=bico;
		} else{
			throw new RuntimeException("Algorithm not found.");
		}


		algorithm.prepareForUse();

		// algorithms.get(a).resetLearningImpl();
		// streams.get(s).restart();

		File resultFile = new File(filename);
		PrintWriter resultWriter = new PrintWriter(resultFile);
		resultWriter.println("points\tsilhouette");

		ArrayList<DataPoint> windowPoints = new ArrayList<DataPoint>(windowSize);

		stream.prepareForUse();

		for (int d = 1; d < length; d++) {
			Instance inst = stream.nextInstance().getData();

			// apparently numAttributes is the class index when no class exists
			if (inst.classIndex() < inst.numAttributes()) {
				inst.deleteAttributeAt(inst.classIndex()); // remove class label
			}
			DataPoint point = new DataPoint(inst, d);
			windowPoints.add(point);

			algorithm.trainOnInstanceImpl(inst);

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

				// windowInstances.clear();
				windowPoints.clear();
				resultWriter.flush();
			}
		}
		resultWriter.close();
	}

}
