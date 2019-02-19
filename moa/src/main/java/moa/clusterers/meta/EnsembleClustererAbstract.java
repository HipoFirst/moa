package moa.clusterers.meta;

import com.github.javacliparser.ListOption;
import com.github.javacliparser.Option;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

public abstract class EnsembleClustererAbstract  extends AbstractClusterer {

	private static final long serialVersionUID = 1L;
	
	// Makes the model slightly more fault tolerant.
	// Sometimes one of the base-models crashes, resulting
	// in the meta-algorithm to crash as well. We ignore
	// it when this happens only occasionally.
	private static final int MAX_TOLLERATED_TRAINING_ERRROS = 100;
	
	private int trainingErrors = 0;
	int instancesSeen = 0;
	protected Clusterer[] ensemble;
		
	public ListOption baselearnersOption = new ListOption("baseClusterer", 'b',
			"The clusterers the ensemble consists of.",
			new ClassOption("learner", ' ', "", Clusterer.class, "clustree.ClusTree"),
			new Option[] {
					new ClassOption("", ' ', "", Clusterer.class, "clustree.ClusTree"),
					new ClassOption("", ' ', "", Clusterer.class, "clustream.WithKmeans") },
			',');

	
	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return null;
	}

	@Override
	public Clustering getClusteringResult() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void resetLearningImpl() {

		this.trainingErrors = 0;
		this.instancesSeen = 0;
		
		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i].resetLearning();
		}		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// train all models
		for (int i = 0; i < this.ensemble.length; i++) {
			try {
				this.ensemble[i].trainOnInstance(inst);
			
			} catch (RuntimeException e) {
				// same strategy as in original BLAST: add fault tolerance in case some individual algorithms crash
				this.trainingErrors += 1;
				if (trainingErrors > MAX_TOLLERATED_TRAINING_ERRROS) {
					throw new RuntimeException(
							"Too much training errors! Latest: " + e.getMessage());
				}
			}
		}
		
		this.instancesSeen++;		
	}
	

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
	}
	
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {


		Option[] learnerOptions = this.baselearnersOption.getList();
		this.ensemble = new Clusterer[learnerOptions.length];
		for (int i = 0; i < learnerOptions.length; i++) {
			monitor.setCurrentActivity("Materializing learner " + (i + 1) + "...",
					-1.0);
			this.ensemble[i] = (Clusterer) ((ClassOption) learnerOptions[i])
					.materializeObject(monitor, repository);
			if (monitor.taskShouldAbort()) {
				return;
			}
			monitor.setCurrentActivity("Preparing learner " + (i + 1) + "...", -1.0);
			this.ensemble[i].prepareForUse(monitor, repository);
			if (monitor.taskShouldAbort()) {
				return;
			}
		}
		super.prepareForUseImpl(monitor, repository);
	}

}
