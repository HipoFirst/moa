package moa.clusterers.meta;


import moa.cluster.Clustering;

public class EnsembleClustererBlast extends EnsembleClustererAbstract{

	private static final long serialVersionUID = 1L;
	
	@Override
	public Clustering getClusteringResult() {
		return this.ensemble[this.bestModel].getClusteringResult(); // TODO we could also return micro clusters here
	}

	
	

}
