package nn;

import java.io.FileNotFoundException;

public class NeuralNetwork {
	
	// The number of inputs into each neuron
	int numInputs=4;
	
	// The number of output units in the output layer
	int numOutputs=3;
	
	// The number of hidden units in the hidden layer
	int numHiddenUnits;
	
	private Layer hiddenLayer;
	
	private Layer outputLayer;
	
	
	public NeuralNetwork(int numHiddenUnits){
		
		this.numHiddenUnits=numHiddenUnits;
		
		this.hiddenLayer= new Layer(numHiddenUnits,numInputs);
		this.outputLayer= new Layer(numOutputs, numHiddenUnits);
		
	}
	
	/*
	 * Function that sets all the weights in the network, starting with the hidden layer, them the output layer
	 * Used when training the neural network to set new values and improve the neural network
	 * EXCLUDING BIAS!!!
	 * */
	public void setAllWeights(double [] weights){
		
		if(weights.length!=(numHiddenUnits)*numInputs+(numOutputs)*numHiddenUnits){
			System.out.println("The number of weights to set is not exact!");
		}
		
			for(int i=0;i<numHiddenUnits;i++){
				for(int j=0;j<numInputs;j++){
				this.hiddenLayer.getNeuron(i).setWeight(j, weights[i*numInputs+j]);
				}
			}
			
			int offset=(numHiddenUnits)*numInputs;
					
			for(int i=0;i<numOutputs;i++){
				for(int j=0;j<numHiddenUnits;j++){
				this.outputLayer.getNeuron(i).setWeight(j, weights[offset+i*numHiddenUnits+j]);
				}
			}
	}
	
public double [] getAllWeights(){
		double [] weights= new double[(numHiddenUnits)*numInputs+(numOutputs)*numHiddenUnits];
			
			for(int i=0;i<numHiddenUnits;i++){
				for(int j=0;j<numInputs;j++){
				weights[i]=this.hiddenLayer.getNeuron(i).getWeight(j);
				}
			}
			
			int offset=(numHiddenUnits)*numInputs;
					
			for(int i=0;i<numOutputs;i++){
				for(int j=0;j<numHiddenUnits;j++){
				weights[offset+i]=this.outputLayer.getNeuron(i).getWeight(j);
				}
			}
			
			return weights;
			
}
	
	/*
	 * Function that generates the output given an input 
	 * 0- Error
	 * 1- Iris setosa
	 * 2- Iris virginica
	 * 3- Iris versicolor
	 * */
	public double [] generateOutput(double [] input){
		
		// check if the length of the input equals numInputs
		if(input.length!=numInputs){
			return null;
		}
		
		// Get the output from the hidden layer first
		double hiddenOutput[]=new double[this.numHiddenUnits];
		
		for(int i=0;i<this.numHiddenUnits;i++){
			
			double value=0;
			double [] w=this.hiddenLayer.getNeuron(i).getWeights();
			
			//System.out.println("hidden layer n: "+this.hiddenLayer.getNeuron(i));
			
			int j=0;
			for(j=0;j<this.numInputs;j++){
				value+=w[j]*input[j];
			}
			
			// add the bias 
			value-=w[j];
			
			// apply the sigmoid function
			double sigVal= (1/(1+Math.pow(Math.E,(value-2*value))));
			
			// add the output to the list
			hiddenOutput[i]=sigVal;
			
			//System.out.println("hiddenOutput["+i+"]="+sigVal);
			
		}
		
		// Now the hiddenOutput is used as input to the output layer
		
		double finalOutput[]= new double[this.numOutputs];
		
		for(int i=0;i<this.numOutputs;i++){
			
			double value=0;
			double [] w=this.outputLayer.getNeuron(i).getWeights();
			
			//System.out.println("hidden layer n: "+this.outputLayer.getNeuron(i));
			
			int j=0;
			for(j=0;j<this.numHiddenUnits;j++){
				value+=w[j]*hiddenOutput[j];
			}
			
			// add the bias 
			value-=w[j];
						
			// apply the sigmoid function
			double sigVal= (1/(1+Math.pow(Math.E,(value-2*value))));
						
			// add the output to the list
			finalOutput[i]=sigVal;
			
			//System.out.println("finalOutput["+i+"]="+sigVal);			
		}
		
		/*
		if(ret==0){
			System.out.println("Error occured, type cannot be determined!");
		}
		else if(ret==1){
			System.out.println("Flower is of type Iris setosa");
		}
		else if(ret==2){
			System.out.println("Flower is of type Iris virginica");
		}
		else if(ret==3){
			System.out.println("Flower is of type Iris versicolor");
		}*/
		
		return finalOutput;
	}
	
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		
		int initNum=6;
		NeuralNetwork nn=new NeuralNetwork(initNum);
		double input []={-2.3d,1.3d,-3.1d,-3.7d};
		//int ret=nn.generateOutput(input);
		
		TSP tsp= new TSP(100,0.85f,0.7f,nn.numInputs*nn.numHiddenUnits+nn.numHiddenUnits*nn.numOutputs,nn);
		
		tsp.createInitialPopulation();
		long start = System.currentTimeMillis();
		//tsp.trainNetwork();
		tsp.trainNetworkDynamic(initNum);
		
		long elapsedTimeMillis = System.currentTimeMillis() - start;
	    // 
	    float elapsedTimeSec = elapsedTimeMillis/1000F;
	    System.out.println("Time elapsed:"+elapsedTimeSec);
	}

}
