package moa.clusterers.meta;

import java.util.ArrayList;

import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.core.AutoExpandVector;
import moa.streams.clustering.RandomRBFGeneratorEvents;

public class ConfStreamMerge extends EnsembleClustererAbstract {

	private static final long serialVersionUID = 1L;

	@Override
	public Clustering getMicroClusteringResult() {
		return this.getEnsembleResult();
	}

	protected Clustering getEnsembleResult() {

		double max = Double.NEGATIVE_INFINITY;
		double min = Double.POSITIVE_INFINITY;
		for(double value: this.silhouettes){
			if(value > max){
				max = value;
			}
			if(value < min){
				min = value;
			}
		}

		double range = max - min;
		for (int i = 0; i < this.silhouettes.size(); i++) {
			this.silhouettes.set(i, (this.silhouettes.get(i) - min) / (range));
		}

		Clustering ensembleClustering = new Clustering(); // create empty clustering

		// get micro clusters for all ensembles
		for (int i = 0; i < this.ensemble.size(); i++) {
			Clustering result = this.ensemble.get(i).clusterer.getMicroClusteringResult();
			if(result == null){
				result = this.ensemble.get(i).clusterer.getClusteringResult();
			}

			AutoExpandVector<Cluster> clusters = result.getClustering();
			// and concatenate them to a single cluster
			for (int j = 0; j < clusters.size(); j++) {
				SphereCluster clstr = (SphereCluster) clusters.get(j); // TODO are there only SphereCluster?
				clstr.setWeight(clstr.getWeight() * this.silhouettes.get(i));
				ensembleClustering.add(clstr);
			}
		}

		return ensembleClustering;
	}

	public static void main(String[] args) {

		ConfStreamMerge algorithm = new ConfStreamMerge();
		RandomRBFGeneratorEvents stream = new RandomRBFGeneratorEvents();
		stream.prepareForUse();
		algorithm.prepareForUse();
		for (int i = 0; i < 1000000; i++) {
			Instance inst = stream.nextInstance().getData();
			algorithm.trainOnInstanceImpl(inst);
		}
		algorithm.getClusteringResult();
	}

}
