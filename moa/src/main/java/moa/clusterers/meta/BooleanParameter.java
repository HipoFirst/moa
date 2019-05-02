package moa.clusterers.meta;

import java.util.ArrayList;

import com.yahoo.labs.samoa.instances.Attribute;

// the representation of a boolean / binary / flag parameter
public class BooleanParameter implements IParameter {
	private String parameter;
	private int numericValue;
	private String value;
	private String[] range = { "false", "true" };
	private Attribute attribute;
	private ArrayList<Double> probabilities;

	public BooleanParameter(BooleanParameter x) {
		this.parameter = x.parameter;
		this.numericValue = x.numericValue;
		this.value = x.value;
		this.range = x.range.clone();
		this.attribute = x.attribute;
		this.probabilities = new ArrayList<Double>(x.probabilities);
	}

	public BooleanParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = String.valueOf(x.value);
		for (int i = 0; i < this.range.length; i++) {
			if (this.range[i].equals(this.value)) {
				this.numericValue = i; // get index of init value
			}
		}
		this.attribute = new Attribute(x.parameter);

		this.probabilities = new ArrayList<Double>(2);
		for (int i = 0; i < 2; i++) {
			this.probabilities.add(0.5); // equal probabilities
		}
	}

	public BooleanParameter copy() {
		return new BooleanParameter(this);
	}

	public String getCLIString() {
		// if option is set
		if (this.numericValue == 1) {
			return ("-" + this.parameter); // only the parameter
		}
		return "";
	}

	public String getCLIValueString() {
		if(this.numericValue == 1){
			return ("");
		} else{
			return(null);
		}
	}


	public double getValue() {
		return this.numericValue;
	}

	public String getParameter() {
		return this.parameter;
	}

	public String[] getRange() {
		return this.range;
	}

	public void sampleNewConfig(int iter, int nbNewConfigurations, int nbVariable) {
		// update configuration
		this.numericValue = EnsembleClustererAbstract.sampleProportionally(this.probabilities);
		String newValue = this.range[this.numericValue];
		System.out.print("Sample new configuration for boolean parameter -" + this.parameter + " with probabilities");
		for (int i = 0; i < this.probabilities.size(); i++) {
			System.out.print(" " + this.probabilities.get(i));
		}
		System.out.println("\t=>\t -" + this.parameter + " " + newValue);
		this.value = newValue;

		// adapt distribution
		// for (int i = 0; i < this.probabilities.size(); i++) {
		// 	this.probabilities.set(i, this.probabilities.get(i) * (1.0 - ((10 - 1.0) / 100)));
		// }
		// this.probabilities.set(this.numericValue, (this.probabilities.get(this.numericValue) + ((10 - 1.0) / 100)));
		this.probabilities.set(this.numericValue, this.probabilities.get(this.numericValue) + (1.0/iter));


		// divide by sum
		double sum = 0.0;
		for (int i = 0; i < this.probabilities.size(); i++) {
			sum += this.probabilities.get(i);
		}
		for (int i = 0; i < this.probabilities.size(); i++) {
			this.probabilities.set(i, this.probabilities.get(i) / sum);
		}
	}
}