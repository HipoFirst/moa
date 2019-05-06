package moa.clusterers.meta;

import com.yahoo.labs.samoa.instances.Attribute;

// the representation of an integer parameter
public class OrdinalParameter implements IParameter {
	private String parameter;
	private String value;
	private int numericValue;
	private String[] range;
	private double std;
	private Attribute attribute;

	// copy constructor
	public OrdinalParameter(OrdinalParameter x) {
		this.parameter = x.parameter;
		this.value = x.value;
		this.numericValue = x.numericValue;
		this.range = x.range.clone();
		this.std = x.std;
		this.attribute = x.attribute;
	}
	
	// init constructor
	public OrdinalParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = String.valueOf(x.value);
		this.range = new String[x.range.length];
		for (int i = 0; i < x.range.length; i++) {
			range[i] = String.valueOf(x.range[i]);
			if (this.range[i].equals(this.value)) {
				this.numericValue = i; // get index of init value
			}
		}
		this.std = (this.range.length - 0) / 8;
		this.attribute = new Attribute(x.parameter);

	}


	public OrdinalParameter copy() {
		return new OrdinalParameter(this);
	}

	public String getCLIString() {
		return ("-" + this.parameter + " " + this.value);
	}

	public String getCLIValueString() {
		return ("" + this.value);
	}

	public double getValue() {
		return this.numericValue;
	}

	public String getParameter() {
		return this.parameter;
	}

	public void sampleNewConfig(int iter, int nbNewConfigurations, int nbVariable) {
		// update configuration
		// treat index of range as integer parameter
		TruncatedNormal trncnormal = new TruncatedNormal(this.numericValue, this.std, (double) (this.range.length - 1),
				0.0); // limits are the indexes of the range
		int newValue = (int) Math.round(trncnormal.sample());
		// System.out.println("Sample new configuration for ordinal parameter -" + this.parameter + " with mean: "
		// 		+ this.numericValue + ", std: " + this.std + ", lb: " + 0 + ", ub: " + (this.range.length - 1)
		// 		+ "\t=>\t -" + this.parameter + " " + this.range[newValue] + " (" + newValue + ")");

		this.numericValue = newValue;
		this.value = this.range[this.numericValue];

		// adapt distribution
		// this.std = this.std * (Math.pow((1.0 / nbNewConfigurations), (1.0 / nbVariable)));
	}

}