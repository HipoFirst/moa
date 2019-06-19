package moa.clusterers.meta;

import java.util.Arrays;

import com.github.javacliparser.Option;
import com.github.javacliparser.Options;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;

import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;
import moa.core.AutoExpandVector;
import moa.options.ClassOption;

public class Algorithm {
	public String algorithm;
	public IParameter[] parameters;
	public AbstractClusterer clusterer;
	public Attribute[] attributes;

	// copy constructor
	public Algorithm(Algorithm x, boolean keepCurrentModel, int verbose) {

		// make a (mostly) deep copy of the algorithm
		this.algorithm = x.algorithm;
		this.attributes = x.attributes; // this is a reference since we dont manipulate the attributes
		this.parameters = new IParameter[x.parameters.length];
		for (int i = 0; i < x.parameters.length; i++) {
			this.parameters[i] = x.parameters[i].copy();
		}
		if (keepCurrentModel) {
			try{
				this.clusterer = (AbstractClusterer) x.clusterer.copy();
			} catch (RuntimeException e){
				if(verbose >= 2){
					System.out.println("Copy failed for " + x.clusterer.getCLICreationString(Clusterer.class) + "! Reinitialise instead.");		
				} 
				this.init();
			}
		}

	}

	// init constructor
	public Algorithm(AlgorithmConfiguration x) {

		this.algorithm = x.algorithm;
		this.parameters = new IParameter[x.parameters.length];

		this.attributes = new Attribute[x.parameters.length];
		for (int i = 0; i < x.parameters.length; i++) {
			ParameterConfiguration paramConfig = x.parameters[i];
			if (paramConfig.type.equals("numeric") || paramConfig.type.equals("float") || paramConfig.type.equals("real")) {
				NumericalParameter param = new NumericalParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter());
			} else if (paramConfig.type.equals("integer")) {
				IntegerParameter param = new IntegerParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter());
			} else if (paramConfig.type.equals("nominal") || paramConfig.type.equals("categorical")) {
				CategoricalParameter param = new CategoricalParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter(), Arrays.asList(param.getRange()));
			} else if (paramConfig.type.equals("boolean") || paramConfig.type.equals("flag")) {
				BooleanParameter param = new BooleanParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter(), Arrays.asList(param.getRange()));
			} else if (paramConfig.type.equals("ordinal")) {
				OrdinalParameter param = new OrdinalParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter());
			} else {
				throw new RuntimeException("Unknown parameter type: '" + paramConfig.type
						+ "'. Available options are 'numeric', 'integer', 'nominal', 'boolean' or 'ordinal'");
			}
		}
		init();
	}

	// initialise a new algorithm using the Command Line Interface (CLI)
	public void init() {
		// construct CLI string from settings, e.g. denstream.WithDBSCAN -e 0.08 -b 0.3
		StringBuilder commandLine = new StringBuilder();
		commandLine.append(this.algorithm); // first the algorithm class
		for (IParameter param : this.parameters) {
			commandLine.append(" ");
			commandLine.append(param.getCLIString());
		}

		// create new clusterer from CLI string
		ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
		this.clusterer = (AbstractClusterer) opt.materializeObject(null, null);
		this.clusterer.prepareForUse();
	}

	// sample a new confguration based on the current one
	public void sampleNewConfig(double lambda, boolean keepCurrentModel, boolean reinitialiseWithMicro, int verbose) {
		// sample new configuration from the parent
		for (IParameter param : this.parameters) {
			param.sampleNewConfig(lambda, verbose);
		}

		if (keepCurrentModel) {
			// Option 1: keep the old state and just change parameter
			StringBuilder commandLine = new StringBuilder();
			for (IParameter param : this.parameters) {
				commandLine.append(param.getCLIString());
			}

			Options opts = this.clusterer.getOptions();
			for (IParameter param : this.parameters) {
				Option opt = opts.getOption(param.getParameter().charAt(0));
				opt.setValueViaCLIString(param.getCLIValueString());
			}

			// these changes do not transfer over directly since all algorithms chache the
			// option values. Therefore we try to adjust the cached values if possible
			try {
				((AbstractClusterer) this.clusterer).adjustParameters();
				if (verbose >= 2) {
					System.out.println("Changed: " + this.clusterer.getCLICreationString(Clusterer.class));
				}
			} catch (UnsupportedOperationException e) {
				if (verbose >= 2) {
					System.out.println("Cannot change parameters of " + this.algorithm + " on the fly, reset instead.");
				}
				keepCurrentModel = false;
			}

		}

		if (!keepCurrentModel) {
			// Option 2: reinitialise the entire state
			Clustering result = this.clusterer.getMicroClusteringResult();
			if(result==null){
				result = this.clusterer.getClusteringResult();
			}

			AutoExpandVector<Cluster> clusters = result.getClusteringCopy();

			this.init();
			if (verbose >= 2) {
				System.out.println("Initialise: " + this.clusterer.getCLICreationString(Clusterer.class));
			}

			if (reinitialiseWithMicro) {
				if (verbose >= 2) {
					System.out.println("Train with existing micro clusters.");
				}
				// train the algorithm with the micro clusters
				for (Cluster cluster : clusters) {
					SphereCluster c = (SphereCluster) cluster; // TODO are there only SphereCluster?
					Instance inst = new DenseInstance(c.getWeight(), c.getCenter());
					this.clusterer.trainOnInstance(inst);
				}
			}

		}
	}

	// returns the parameter values as an array
	public double[] getParamVector(int padding) {
		double[] params = new double[this.parameters.length + padding];
		int pos = 0;
		for (IParameter param : this.parameters) {
			params[pos++] = param.getValue();
		}
		return params;
	}
}