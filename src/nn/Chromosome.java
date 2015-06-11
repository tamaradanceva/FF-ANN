package nn;

import java.util.Arrays;

public class Chromosome implements Comparable<Chromosome>{
	
	private int id;
	
	private double fitness;
	
	// This is an array made up of all the weights in the neural network
	private double [] weights;
	
	public Chromosome(int id,double fitness,double [] weights){
		this.id=id;
		this.fitness=fitness;
		this.weights=new double[weights.length];
		for(int i=0;i<weights.length;i++){
			this.weights[i]=weights[i];
		}
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public double getFitness() {
		return fitness;
	}

	public void setFitness(double fitness) {
		this.fitness = fitness;
	}

	public double[] getWeights() {
		return weights;
	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}
	
	public void setWeight(int i,double weight){
		this.weights[i]=weight;
	}

	public double getWeight(int i){
		return this.weights[i];
	}
	@Override
	public String toString() {
		return "Chromosome [id=" + id + ", fitness=" + fitness + ", weights="
				+ Arrays.toString(weights) + "]";
	}
	
	
	@Override
	public int compareTo(Chromosome arg0) {
		double d=this.fitness-arg0.fitness;
		int res=0;
		if(d>0) res=1;
		if(d==0) res=0;
		if(d<0) res=-1;
		return res;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(weights);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Chromosome other = (Chromosome) obj;
		if (!Arrays.equals(weights, other.weights))
			return false;
		return true;
	}
	
	
	
	

}
