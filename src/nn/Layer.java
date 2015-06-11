package nn;

import java.util.Arrays;

public class Layer {
	
	private Neuron[] neurons;
	
	public Layer(int numNeurons, int numInputs){
		
		this.neurons=new Neuron[numNeurons];
		
		for(int i=0;i<numNeurons;i++){
			this.neurons[i]=new Neuron(numInputs);
		}
	}
	
	public Neuron getNeuron(int i){
		return neurons[i];
	}

	@Override
	public String toString() {
		return "Layer [neurons=" + Arrays.toString(neurons) + "]";
	}
	

}
