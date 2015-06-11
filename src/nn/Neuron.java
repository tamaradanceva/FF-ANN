package nn;

import java.util.Arrays;
import java.util.Random;

public class Neuron {
	
	private double [] weights;
	
	public Neuron(int numInputs){
		
		Random r=new Random();
		
		// One extra weight is used for the bias, bias will be the last value
		this.weights=new double [numInputs+1];
		
		for(int i=0;i<numInputs+1;i++){
			this.weights[i]=r.nextDouble() * (0.6d - (-0.6d)) + (-0.6d);
		}
		
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double[] weights) {
		for(int i=0;i<weights.length;i++){
			this.weights[i]=weights[i];
		}
	}
	
	public void setWeight(int i, double weight){
		this.weights[i]=weight;
	}

	public double getWeight(int i){
		return this.weights[i];
	}
	
	@Override
	public String toString() {
		return "Neuron [weights=" + Arrays.toString(weights) + "]";
	}
	
	
	

}
