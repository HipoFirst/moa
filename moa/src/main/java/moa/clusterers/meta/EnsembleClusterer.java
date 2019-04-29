package moa.clusterers.meta;

import java.util.ArrayList;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.core.AutoExpandVector;

public class EnsembleClusterer extends EnsembleClustererAbstract{

	private static final long serialVersionUID = 1L;

	@Override
	public Clustering getMicroClusteringResult() {	
		return this.getEnsembleResult();
	}

	protected Clustering getEnsembleResult() {
		ArrayList<Double> silhs = this.silhouette.getAllValues(0);

		// normalize, could also use DoubleVector.normalize()
		double min = this.silhouette.getMinValue(0);
		double max = this.silhouette.getMaxValue(0);
		double range = max - min;
		for(int i=0; i < silhs.size(); i++){
			silhs.set(i, (silhs.get(i) - min) / (range));
		}
		
		Clustering ensembleClustering = new Clustering(); // create empty clustering
		
		// get micro clusters for all ensembles
		for (int i = 0; i < this.ensemble.size(); i++) {
			Clustering clustering = this.ensemble.get(i).clusterer.getMicroClusteringResult();
			AutoExpandVector<Cluster> clusters = clustering.getClustering(); 
			// and concatenate them to a single cluster
			for (int j = 0; j < clusters.size(); j++) {
				SphereCluster clstr = (SphereCluster) clusters.get(j); // TODO are there only SphereCluster?
				clstr.setWeight(clstr.getWeight() * silhs.get(j));
				ensembleClustering.add(clstr); 
			}
		}
		
		return ensembleClustering;
	}
		
	

}
