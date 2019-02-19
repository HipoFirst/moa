package moa.clusterers.meta;

import java.util.ArrayList;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Clustering;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;

public class EnsembleClustererBlast extends EnsembleClustererAbstract{

	private static final long serialVersionUID = 1L;
	
	public IntOption windowSizeOption = new IntOption("windowSize", 'w',
			"The window size over which Online Performance Estimation is done.", 1000,
			1, Integer.MAX_VALUE);

	int windowsize = 1000; // number of points to evaluate
	int activeModel = -1;
	ArrayList<DataPoint> windowPoints = new ArrayList<DataPoint>(windowsize); // points to evaluate
 

	@Override
	public Clustering getClusteringResult() {
		if(this.instancesSeen < this.windowsize) findBestModel(); // if called before the end of first window
		
		return this.ensemble[this.activeModel].getClusteringResult();
	}

	@Override
	public void resetLearningImpl() {
		super.resetLearning();
		
		this.windowPoints = new ArrayList<DataPoint>(windowsize);
		this.activeModel = -1;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		DataPoint point = new DataPoint(inst,instancesSeen); // create data points from instance
		this.windowPoints.add(point); // remember points of the current window
		
		super.trainOnInstance(inst);
		
		// when window is complete, evaluate and find the best clustering result
		if (this.instancesSeen % this.windowsize == 0) {
			findBestModel();
			windowPoints.clear(); // flush the current window
		}
	}
	
	
	protected void findBestModel() {
		double maxVal = -1*Double.MAX_VALUE;
		
			for (int i = 0; i < this.ensemble.length; i++) {
				 // get current macro clusters
				Clustering result = this.ensemble[i].getClusteringResult();
				
				 // init evaluation measure TODO make this a user parameter
				SilhouetteCoefficient silh = new SilhouetteCoefficient();
				silh.evaluateClustering(result, null, windowPoints); // compute evaluation measure
				double performance = silh.getLastValue(0);
				System.out.println(performance);
				
				// find best clustering result
				if(performance > maxVal){
					maxVal = performance;
					this.activeModel = i;
				}
			}
	}
		


	

}
