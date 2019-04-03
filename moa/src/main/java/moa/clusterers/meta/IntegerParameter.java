package moa.clusterers.meta;

import com.yahoo.labs.samoa.instances.Attribute;

// the representation of an integer parameter
public class IntegerParameter implements IParameter {
	private String parameter;
	private int value;
	private int[] range;
	private double std;
	private Attribute attribute;

	public IntegerParameter(IntegerParameter x) {
		this.parameter = x.parameter;
		this.value = x.value;
		this.range = x.range.clone();
		this.std = x.std;
		this.attribute = x.attribute;// new Attribute(x.parameter);
	}

	public IntegerParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = (int) (double) x.value; // TODO fix casts
		this.range = new int[x.range.length];
		for (int i = 0; i < x.range.length; i++) {
			range[i] = (int) (double) x.range[i];
		}
		this.std = (this.range[1] - this.range[0]) / 2;
		this.attribute = new Attribute(x.parameter);
	}

	public IntegerParameter copy() {
		return new IntegerParameter(this);
	}

	public String getCLIString() {
		return ("-" + this.parameter + " " + this.value);
	}

	public double getValue() {
		return this.value;
	}

	public String getParameter() {
		return this.parameter;
	}

	public void sampleNewConfig(int nbNewConfigurations, int nbVariable) {
		// update configuration
		// for integer features use truncated normal distribution
		TruncatedNormal trncnormal = new TruncatedNormal(this.value, this.std, this.range[0], this.range[1]);
		int newValue = (int) Math.round(trncnormal.sample());
		System.out.println("Sample new configuration for integer parameter -" + this.parameter + " with mean: "
				+ this.value + ", std: " + this.std + ", lb: " + this.range[0] + ", ub: " + this.range[1] + "\t=>\t -"
				+ this.parameter + " " + newValue);

		this.value = newValue;
		// adapt distribution
		this.std = this.std * (Math.pow((1.0 / nbNewConfigurations), (1.0 / nbVariable)));
	}
}