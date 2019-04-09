package moa.clusterers.meta;


import moa.cluster.Clustering;

public class EnsembleClustererBlast extends EnsembleClustererAbstract{

	private static final long serialVersionUID = 1L;
	
	@Override
	public Clustering getMicroClusteringResult() {
		return this.ensemble.get(this.bestModel).clusterer.getMicroClusteringResult(); // TODO we could also return micro clusters here
	}

	
	

}
