package moa.clusterers.meta;

import java.util.Arrays;

import com.yahoo.labs.samoa.instances.Attribute;

import moa.clusterers.Clusterer;
import moa.options.ClassOption;

public class Algorithm {
	public String algorithm;
	public IParameter[] parameters;
	public Clusterer clusterer;
	public Attribute[] attributes;

	// copy constructor
	public Algorithm(Algorithm x) {

		// make a (mostly) deep copy of the algorithm
		this.algorithm = x.algorithm;
		this.attributes = x.attributes; // this is a reference since we dont manipulate the attributes
		this.parameters = new IParameter[x.parameters.length];
		for (int i=0; i<x.parameters.length; i++){
			this.parameters[i] = x.parameters[i].copy();
		}
		// this.clusterer = x.clusterer.copy();
		
		// init(); // we dont initialise here because we want to manipulate the
		// parameters first
	}

	// init constructor
	public Algorithm(AlgorithmConfiguration x) {

		this.algorithm = x.algorithm;
		this.parameters = new IParameter[x.parameters.length];

		this.attributes = new Attribute[x.parameters.length];
		for (int i=0; i<x.parameters.length; i++) {
			ParameterConfiguration paramConfig = x.parameters[i];
			if (paramConfig.type.equals("numeric")) {
				NumericalParameter param = new NumericalParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter());
			} else if (paramConfig.type.equals("integer")) {
				IntegerParameter param = new IntegerParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter());
			} else if (paramConfig.type.equals("nominal")) {
				CategoricalParameter param = new CategoricalParameter(paramConfig);
				this.parameters[i] = param;
				this.attributes[i] = new Attribute(param.getParameter(), Arrays.asList(param.getRange()));
			} else if (paramConfig.type.equals("boolean")) {
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

	public void init() {
		// initialise a new algorithm using the Command Line Interface (CLI)
		// construct CLI string from settings, e.g. denstream.WithDBSCAN -e 0.08 -b 0.3
		StringBuilder commandLine = new StringBuilder();
		commandLine.append(this.algorithm); // first the algorithm class
		for (IParameter param : this.parameters) {
			commandLine.append(" ");
			commandLine.append(param.getCLIString());
		}
		System.out.println("Initialise: " + commandLine.toString());

		// create new clusterer from CLI string
		ClassOption opt = new ClassOption("", ' ', "", Clusterer.class, commandLine.toString());
		this.clusterer = (Clusterer) opt.materializeObject(null, null);
		this.clusterer.prepareForUse();
	}

	public void sampleNewConfig(int iter, int nbNewConfigurations) {
		// sample new configuration from the parent
		for (IParameter param : this.parameters) {
			param.sampleNewConfig(iter, nbNewConfigurations, this.parameters.length);
		}
	}

	public double[] getParamVector(int padding) {
		double[] params = new double[this.parameters.length + padding];
		int pos = 0;
		for (IParameter param : this.parameters) {
			params[pos++] = param.getValue();
		}
		return params;
	}
}