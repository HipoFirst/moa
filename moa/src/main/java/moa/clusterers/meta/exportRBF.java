package moa.clusterers.meta;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

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
import moa.streams.clustering.ClusteringStream;
import moa.streams.clustering.RandomRBFGeneratorEvents;
import moa.streams.clustering.SimpleCSVStream;


public abstract class exportRBF{


	public static void main(String[] args) throws FileNotFoundException {

		RandomRBFGeneratorEvents rbf = new RandomRBFGeneratorEvents();
		rbf.modelRandomSeedOption.setValue(2410);
		rbf.eventFrequencyOption.setValue(30000);
		rbf.eventDeleteCreateOption.setValue(true);
		rbf.eventMergeSplitOption.setValue(true);
		rbf.prepareForUse();


		File file = new File("RBF.csv");
		PrintWriter writer = new PrintWriter(file);

		for(int i = 0; i < 2000000; i++){
			Instance inst = rbf.nextInstance().getData();
			double[] val = inst.toDoubleArray();

			int attrs = inst.numAttributes() - 1; // last is class
			for(int j = 0; j < attrs ; j++){
				writer.print(val[j]);
				if(j != attrs-1){
					writer.print(",");
				}
			}
			writer.print("\n");
		}
		writer.close();
	}
}
