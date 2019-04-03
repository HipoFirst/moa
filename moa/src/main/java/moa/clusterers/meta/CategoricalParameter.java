package moa.clusterers.meta;

import java.util.ArrayList;
import java.util.Arrays;

import com.yahoo.labs.samoa.instances.Attribute;

// the representation of a categorical / nominal parameter
public class CategoricalParameter implements IParameter {
	private String parameter;
	private int numericValue;
	private String value;
	private String[] range;
	private Attribute attribute;
	private ArrayList<Double> probabilities;

	public CategoricalParameter(CategoricalParameter x) {
		this.parameter = x.parameter;
		this.numericValue = x.numericValue;
		this.value = x.value;
		this.range = x.range.clone();
		this.attribute = x.attribute;
		this.probabilities = new ArrayList<Double>(x.probabilities);
	}

	public CategoricalParameter(ParameterConfiguration x) {
		this.parameter = x.parameter;
		this.value = String.valueOf(x.value);
		this.range = new String[x.range.length];
		for (int i = 0; i < x.range.length; i++) {
			range[i] = String.valueOf(x.range[i]);
			if (this.range[i].equals(this.value)) {
				this.numericValue = i; // get index of init value
			}
		}
		this.attribute = new Attribute(x.parameter, Arrays.asList(range));
		this.probabilities = new ArrayList<Double>(x.range.length);
		for (int i = 0; i < x.range.length; i++) {
			this.probabilities.add(1.0 / x.range.length); // equal probabilities
		}
	}

	public CategoricalParameter copy() {
		return new CategoricalParameter(this);
	}

	public String getCLIString() {
		return ("-" + this.parameter + " " + this.value);
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

	public void sampleNewConfig(int nbNewConfigurations, int nbVariable) {
		// update configuration
		this.numericValue = EnsembleClustererAbstract.sampleProportionally(this.probabilities);
		String newValue = this.range[this.numericValue];

		System.out.print("Sample new configuration for nominal parameter -" + this.parameter + "with probabilities");
		for (int i = 0; i < this.probabilities.size(); i++) {
			System.out.print(" " + this.probabilities.get(i));
		}
		System.out.println("\t=>\t -" + this.parameter + " " + newValue);
		this.value = newValue;

		// adapt distribution
		for (int i = 0; i < this.probabilities.size(); i++) {
			// TODO not directly transferable, (1-((iter -1) / maxIter))
			this.probabilities.set(i, this.probabilities.get(i) * (1.0 - ((10 - 1.0) / 100)));
		}
		this.probabilities.set(this.numericValue, (this.probabilities.get(this.numericValue) + ((10 - 1.0) / 100)));

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