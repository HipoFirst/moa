package moa.clusterers.meta;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.core.AutoExpandVector;

public class EnsembleClusterer extends EnsembleClustererAbstract{

	private static final long serialVersionUID = 1L;

	@Override
	public Clustering getClusteringResult() {	
		return this.getEnsembleResult();
	}

	protected Clustering getEnsembleResult() {
		Clustering ensembleClustering = new Clustering(); // create empty clustering
		
		// get micro clusters for all ensembles
		for (int i = 0; i < this.ensemble.length; i++) {
			Clustering clustering = this.ensemble[i].getMicroClusteringResult();
			System.out.println(clustering.size());
			AutoExpandVector<Cluster> clusters = clustering.getClustering(); 
			// and concatenate them to a single cluster
			for (int j = 0; j < clusters.size(); j++) {
				ensembleClustering.add(clusters.get(j)); 
			}
		}
		System.out.println("Combined: " + ensembleClustering.size());
		
		return ensembleClustering;
	}
		
	

}
