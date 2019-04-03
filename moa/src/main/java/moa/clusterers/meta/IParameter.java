package moa.clusterers.meta;

// interface allows us to maintain a single list of parameters
public interface IParameter {
	public void sampleNewConfig(int nbNewConfigurations, int nbVariable);

	public IParameter copy();

	public String getCLIString();

	public double getValue();

	public String getParameter();
}
