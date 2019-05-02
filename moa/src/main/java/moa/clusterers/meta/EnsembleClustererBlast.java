package moa.clusterers.meta;

import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Clustering;
import moa.streams.clustering.RandomRBFGeneratorEvents;

public class EnsembleClustererBlast extends EnsembleClustererAbstract {

	private static final long serialVersionUID = 1L;

	@Override
	public Clustering getMicroClusteringResult() {
		return this.ensemble.get(this.bestModel).clusterer.getMicroClusteringResult(); // TODO we could also return
																						// micro clusters here
	}

	public static void main(String[] args) {

		EnsembleClustererBlast algorithm = new EnsembleClustererBlast();
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
