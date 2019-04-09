package moa.clusterers.meta;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.core.AutoExpandVector;

public class EnsembleClusterer extends EnsembleClustererAbstract{

	private static final long serialVersionUID = 1L;

	@Override
	public Clustering getMicroClusteringResult() {	
		return this.getEnsembleResult();
	}

	protected Clustering getEnsembleResult() {
		Clustering ensembleClustering = new Clustering(); // create empty clustering
		
		// get micro clusters for all ensembles
		for (int i = 0; i < this.ensemble.size(); i++) {
			Clustering clustering = this.ensemble.get(i).clusterer.getMicroClusteringResult();
			AutoExpandVector<Cluster> clusters = clustering.getClustering(); 
			// and concatenate them to a single cluster
			for (int j = 0; j < clusters.size(); j++) {
				ensembleClustering.add(clusters.get(j)); 
			}
		}
		
		return ensembleClustering;
	}
		
	

}
