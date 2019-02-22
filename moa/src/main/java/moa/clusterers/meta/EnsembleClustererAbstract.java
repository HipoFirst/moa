package moa.clusterers.meta;


import java.util.ArrayList;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.ListOption;
import com.github.javacliparser.Option;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.meta.AdaptiveRandomForestRegressor;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;
import moa.clusterers.denstream.WithDBSCAN;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;
import moa.options.ClassOption;
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
			new ClassOption("learner", ' ', "", Clusterer.class, "denstream.WithDBSCAN"),
			new Option[] {
					new ClassOption("", ' ', "", Clusterer.class, "denstream.WithDBSCAN -e 0.06"),
					new ClassOption("", ' ', "", Clusterer.class, "denstream.WithDBSCAN -e 0.05") },
			',');
	
	public IntOption windowSizeOption = new IntOption("windowSize", 'w',
			"The window size over which Online Performance Estimation is done.", 1000,
			1, Integer.MAX_VALUE);

	public FileOption fileOption = new FileOption("ConfigurationFile", 'f', "Configuration file in json format.",
            "", "json", false);
	
	
    public void init() {
    	this.fileOption.getFile();
    }
	
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
			System.out.println("--------------");
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
				// compute silhouette width
				silh.evaluateClustering(result, null, windowPoints);
				double performance = silh.getLastValue(0);
				System.out.println("Result -e=" + ((WithDBSCAN)this.ensemble[i]).epsilonOption.getValue() + ":\t" + performance);
				
				// find best clustering result
				if(performance > maxVal){
					maxVal = performance;
					this.bestModel = i; // the clusterer with the best result becomes the active one
				}
				
				
//				Option[] learnerOptions = this.baselearnersOption.getList();
//				System.out.println(this.ensemble[1].getOptions()); // get all available options
//				System.out.println(learnerOptions[0].getValueAsCLIString());
//				System.out.println(learnerOptions[0].getStateString());
//				System.out.println(learnerOptions[0].getDefaultCLIString());
//				System.out.println(learnerOptions[0].getCLIChar());
				
				// unpack CLI string into params and class
//				String cliString = this.ensemble[0].getCLICreationString(Clusterer.class);
//
//		        cliString = cliString.trim();
//		        int firstSpaceIndex = cliString.indexOf(' ', 0);
//		        String className;
//		        ArrayList<String> classOptions = new ArrayList<String>();
//		        ArrayList<String> classOptionValues = new ArrayList<String>();
//
//		        if (firstSpaceIndex > 0) {
//		            className = cliString.substring(0, firstSpaceIndex); // get class name
//		            int index = cliString.indexOf("-", firstSpaceIndex+1);
//		            while (index >= 0) {
//		                int nextIndex = cliString.indexOf("-", index + 1);
//		                if(nextIndex < 0) {
//			                String test = cliString.substring(index+1, cliString.length()).trim());
//		                } else {
//			                classOptions.add(cliString.substring(index+1, nextIndex).trim());
//		                }
//		                index = nextIndex;
//		            }
//		        } else {
//		            className = cliString;
//		        }

				
				
//				String cliString = this.ensemble[0].getCLICreationString(Clusterer.class);
//
//		        cliString = cliString.trim();
//		        int firstSpaceIndex = cliString.indexOf(' ', 0);
//		        String className;
//		        if (firstSpaceIndex > 0) {
//			        cliString = cliString.substring(firstSpaceIndex, cliString.length()).trim();
//
//		        	String[] options = cliString.split("-");
//		        	for(int j =0; j < options.length; j++) {
//			        	String[] option = cliString.trim().split(" ");
//			        	String opt = option[0];
//			        	String val = option[1];
//		        	}
//		        } else {
//		            className = cliString;
//		        }
//				
						
				// create new instance based on settings and performance
				double[] values = {((WithDBSCAN)this.ensemble[i]).epsilonOption.getValue(), performance};
		        Instance inst = new DenseInstance(1.0, values);
		        
		        // parameter settings are the attributes
		        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		        for (int j = 0; j < 1; j++) {
		            attributes.add(new Attribute("att" + (j + 1)));
		        }
	            attributes.add(new Attribute("class1"));
		        	
				// add header
		        Instances dataset = new Instances(null,	attributes, 0);
		        dataset.setClassIndex(dataset.numAttributes() - 1);

		        inst.setDataset(dataset);

				// train adaptive random forest regressor based on performance of model
				this.ARFregs[i].trainOnInstanceImpl(inst);
				
				
				// predict performance of new configuration
				double[] vals = {0.02};
				Instance newInst = new DenseInstance(1.0, vals);
		        Instances noClassDataset = new Instances(null, attributes, 0);
		        noClassDataset.setClassIndex(noClassDataset.numAttributes() - 1); // TODO why is this even necessary, class does not exist
		        newInst.setDataset(noClassDataset);
		        
				double prediction = this.ARFregs[i].getVotesForInstance(newInst)[0];
				System.out.println("-> Prediction -e=" +  vals[0] + ":\t"+ prediction);
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
		algorithm.getClusteringResult();
		
		System.out.println("-------------");
		
		EnsembleClusterer algorithm2 = new EnsembleClusterer();
		RandomRBFGeneratorEvents stream2 = new RandomRBFGeneratorEvents();
		stream2.prepareForUse();
		algorithm2.prepareForUse();
		for(int i=0; i<100000; i++) {
			Instance inst = stream2.nextInstance().getData();
			algorithm2.trainOnInstanceImpl(inst);
		}
		algorithm2.getClusteringResult();
		
	}
	
}



