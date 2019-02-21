package moa.clusterers.meta;

import java.util.ArrayList;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.ListOption;
import com.github.javacliparser.Option;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;
import moa.options.ClassOption;
import moa.streams.InstanceStream;
import moa.streams.clustering.RandomRBFGeneratorEvents;
import moa.tasks.TaskMonitor;

public abstract class EnsembleClustererAbstract extends AbstractClusterer {

	private static final long serialVersionUID = 1L;

	int instancesSeen;
	protected Clusterer[] ensemble;
	int bestModel;
	ArrayList<DataPoint> windowPoints;
	AdaptiveRandomForestRegressor[] ARFregs;
		
	public ListOption baselearnersOption = new ListOption("baseClusterer", 'b',
			"The clusterers the ensemble consists of.",
			new ClassOption("learner", ' ', "", Clusterer.class, "clustree.ClusTree"),
			new Option[] {
					new ClassOption("", ' ', "", Clusterer.class, "clustree.ClusTree"),
					new ClassOption("", ' ', "", Clusterer.class, "clustream.WithKmeans") },
			',');
	
	public IntOption windowSizeOption = new IntOption("windowSize", 'w',
			"The window size over which Online Performance Estimation is done.", 1000,
			1, Integer.MAX_VALUE);

	
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

		this.instancesSeen = 0;
		this.bestModel = 0;
		this.windowPoints = new ArrayList<DataPoint>(windowSizeOption.getValue());
		this.ARFregs = new AdaptiveRandomForestRegressor[this.ensemble.length];

		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i].resetLearning();
			this.ARFregs[i] = new AdaptiveRandomForestRegressor(); // one regressor per configuration
			this.ARFregs[i].prepareForUse();
		}		
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {		
		DataPoint point = new DataPoint(inst,instancesSeen); // create data points from instance
		this.windowPoints.add(point); // remember points of the current window
	
		// train all models
		for (int i = 0; i < this.ensemble.length; i++) {
			this.ensemble[i].trainOnInstance(inst);
		}
		
		// every windowSize we update the configurations
		if (this.instancesSeen % windowSizeOption.getValue() == 0) {
			updateConfiguration();	
			windowPoints.clear(); // flush the current window
		}
		
		this.instancesSeen++;		
	}
	
	protected void updateConfiguration() {
		// init evaluation measure TODO make this a user parameter
		SilhouetteCoefficient silh = new SilhouetteCoefficient();

		double maxVal = -1*Double.MAX_VALUE;
		for (int i = 0; i < this.ensemble.length; i++) {
			 // get current macro clusters
			Clustering result = this.ensemble[i].getClusteringResult();
			
			// evaluate clustering if we have one
			// if no algorithm produces a clustering, the old clusterer will remain active			
			if(result != null) {
				silh.evaluateClustering(result, null, windowPoints); // compute evaluation measure, true clustering not needed for silouette
				double performance = silh.getLastValue(0);
				System.out.println(performance);
				
				// find best clustering result
				if(performance > maxVal){
					maxVal = performance;
					this.bestModel = i; // the clusterer with the best result becomes the active one
				}
				
				
				// train adaptive random forest based on performance of model
				double[] values = {1,2,3, performance}; // create new instance based on settings and performance
		        Instance inst = new DenseInstance(1.0, values);
		        
		        // add header
		        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		        for (int j = 0; j < 3; j++) {
		            attributes.add(new Attribute("att" + (j + 1)));
		        }
		        
		        // performance measure is the "class"
		        ArrayList<String> classLabels = new ArrayList<String>();
		        for (int j = 0; j < 1; j++) {
		            classLabels.add("class" + (j + 1));
		        }	        
		        
		        Instances dataset = new Instances(null,	attributes, 0);
		        inst.setDataset(dataset);

				this.ARFregs[i].trainOnInstanceImpl(inst);
			}
		}
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

	
	
	
	
	
	
	
	public static void main(String [ ] args){
		EnsembleClustererBlast algorithm = new EnsembleClustererBlast();
		RandomRBFGeneratorEvents stream = new RandomRBFGeneratorEvents();
		stream.prepareForUse();
		algorithm.prepareForUse();
		for(int i=0; i<100000; i++) {
			Instance inst = stream.nextInstance().getData();
			algorithm.trainOnInstanceImpl(inst);
		}
		Clustering cluster = algorithm.getClusteringResult();
		
		System.out.println("-------------");
		
		EnsembleClusterer algorithm2 = new EnsembleClusterer();
		RandomRBFGeneratorEvents stream2 = new RandomRBFGeneratorEvents();
		stream2.prepareForUse();
		algorithm2.prepareForUse();
		for(int i=0; i<100000; i++) {
			Instance inst = stream2.nextInstance().getData();
			algorithm2.trainOnInstanceImpl(inst);
		}
		Clustering cluster2 = algorithm2.getClusteringResult();
		
	}
	
}



