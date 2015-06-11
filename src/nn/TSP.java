package nn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class TSP {
	
	// Number of routes in the population
		private int populationSize;
		
		// List of the populations' chromosomes
		private List<Chromosome> population;
		
		// Number of routes in a sample taken when selecting, using roulette wheel
		private int sampleSizeSelection;
		
		// Number of chromosomes(routes) to copy into the next generation before crossover
		private int numOfBestToKeep;

		// True for roulette wheel, false for tournament selection
		private boolean typeSelection;
		
		// if typesel=false, prob that the better will be selected
		private float tournamentBestProb;
		
		// Probability that the selected routes will crossover
		private float crossoverProb;
		// True for cycle crossover, false for order crossover
		private boolean typeCrosssover;
		
		// Probability that the selected route will mutate
		private float muatationProb;
		
		// Best route found so far
		private Chromosome bestChromosome;
		
		// Number of iterations (new populations created)
		private int numOfIterations;
		
		// M A Y B E
		// count of Number of iterations same (new populations created) 
		private int numOfIterationsSame;
		//Number of the iteration that last had the same route
		private int numOfI;
		// num of iterations as stopping condition
		private int stopNumOfIterations;
		
		private Test [] tests=new Test[150];
		
		private int numWeights;
		
		private NeuralNetwork nn;
		
		private List<Double> averageTrainingMSE;
		
		private List<Double> averageValidationMSE;
		
		private double mseTrain;
		
		//private double averageTrainingMSE;
		
		//private double averageValidationMSE;
		
		
		public TSP(int populationSize, float crossoverProb, float mutationProb, int numWeights, NeuralNetwork nn ){
			this.populationSize=populationSize;
			this.crossoverProb=crossoverProb;
			this.muatationProb=mutationProb;
			this.numWeights=numWeights;
			this.nn=nn;
			
			this.averageTrainingMSE=new ArrayList<Double>();
			this.averageValidationMSE=new ArrayList<Double>();
			
			this.sampleSizeSelection=Math.round(this.populationSize/(1.5f));
			this.numOfBestToKeep=this.sampleSizeSelection/5;
			if(this.numOfBestToKeep%2==1){
				this.numOfBestToKeep++;
			}
			// roulette true, tournament false
			this.typeSelection=true;
			
			this.typeCrosssover=false;
			
			this.stopNumOfIterations=500;
			this.numOfIterationsSame=0;
			this.numOfI=-1;
			
			initTest();
			// I M P L E M E N T  N O R M A L I Z A T I O N
			normalizeDataSet();
			
		}
		
		
		public void initTest(){
			int counter=0;
			File file= new File("IrisDataSet.txt");
			try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			    String line;
			    String [] params;
			    while ((line = br.readLine()) != null) {
			       // process the line.
			    	//System.out.println(line);
			    	params=line.split(" ");
			    	double p []=new double[4];
			    	p[0]=Double.parseDouble(params[0]);
			    	p[1]=Double.parseDouble(params[1]);
			    	p[2]=Double.parseDouble(params[2]);
			    	p[3]=Double.parseDouble(params[3]);
			    	int r=0;
			    	if(params[4].contains("I.setosa")){
			    		r=1;
			    	}
			    	else if(params[4].contains("I.versicolor")){
			    		r=2;
			    	}
			    	else if(params[4].contains("I.virginica")){
			    		r=3;
			    	}
			    	tests[counter]=new Test(p,r);
			    //	System.out.println(tests[counter]);
			    	counter++;
			    }
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			//System.out.println("Number of lines:"+counter);
		}
		
		double getMean(double [] data)
	    {
	        double sum = 0.0;
	        for(double a : data)
	            sum += a;
	        return sum/data.length;
	    }

	    double getVariance(double [] data)
	    {
	        double mean = getMean(data);
	        double temp = 0;
	        for(double a :data)
	            temp += (mean-a)*(mean-a);
	        return temp/data.length;
	    }
		
		public void normalizeDataSet(){
			
			for(int i=0;i<4;i++){
				double [] data= new double[tests.length];
				double min=0d;
				double max=0d;
				for(int j=0;j<tests.length;j++){
					data[j]=tests[j].getParam(i);
					if(j==0){
						min=data[j];
						max=data[j];
					}
					else {
						if(data[j]<min){ min=data[j];}
						if(data[j]>max){ max=data[j];}
					}
				}
				double mean=getMean(data);
				double variance=getVariance(data);
				
				for(int j=0;j<tests.length;j++){
					tests[j].setParam(i,(data[j]-min)/(max-min));
					//tests[j].setParam(i,(data[j]-mean)/variance);
				}
				
			}
			
		}
		
		public void createInitialPopulation(){
			
			this.numWeights=nn.numInputs*nn.numHiddenUnits+nn.numHiddenUnits*nn.numOutputs;
		
			this.population=new ArrayList<Chromosome>();
			Random r=new Random();
			
			double averageTMSE=0d;
			
			double averageVMSE=0d;
			
			for(int i=0;i<populationSize;i++){
				
				double w []=new double[numWeights];
				
				for(int j=0;j<numWeights;j++){
					w[j]=r.nextDouble() * (0.6d - (-0.6d)) + (-0.6d);
				}
				
				// Calculate fitness
				double fitness=calculateFitness(w);
				
				// Training set mse
				averageTMSE+=this.mseTrain;
				averageVMSE+=validate(w);
		
				Chromosome c= new Chromosome(i, fitness, w);
				//System.out.println(c);
				this.population.add(c);
				
				
				
				
				
				if(i==0){
					bestChromosome= new Chromosome(c.getId(),c.getFitness(),c.getWeights());
				}
				else {
					if(c.getFitness()<bestChromosome.getFitness()){
						bestChromosome= new Chromosome(c.getId(),c.getFitness(),c.getWeights());
					}
				}
			}// end for all population
			
			averageTMSE/=populationSize;
			averageVMSE/=populationSize;
			
			this.averageTrainingMSE.add(averageTMSE);
			this.averageValidationMSE.add(averageVMSE);
			
			//System.out.println("BEST CHROMOSOME:"+bestChromosome);
			
			
		}
		
		/*
		 * Function that calculates the fitness of a chromosome on the training set
		 * */
		public double calculateFitness(double [] weights){
			
			// Set chromosome's random generated weights to the network to test its perg on the whole training set
			nn.setAllWeights(weights);
			
			double mse=0d;
			
			int testOffset=0;
			int cor=0;
			//Now test over the training set
			for(int i=0;i<90;i++){
				if(i==30) testOffset=50;
				if(i==60) testOffset=100;
				
				double [] res=nn.generateOutput(tests[testOffset+(i%30)].getParams());
				
				//System.out.println("Test:"+tests[testOffset+i]+" ; RES:"+res[0]+" "+res[1]+" "+res[2]);
				int indMax=0;
				double max=0d;
				for(int j=0;j<res.length;j++){
					if (j==0){max=res[j];indMax=1;}
					else {
						if(res[j]>max){
							max=res[j];
							indMax=j+1;
						}
					}
				}
				
				if(indMax!=tests[testOffset+(i%30)].getCorrectResult()){
					mse+=Math.pow(max-res[tests[testOffset+(i%30)].getCorrectResult()-1]+0.1,2);
					cor++;
				}
				
				
				
			}
			
			mse=mse/90;
			this.mseTrain=mse;
			
			return cor;
		}
		
		public double validate(double [] weights){
			
			// Set chromosome's random generated weights to the network to test its perg on the whole training set
						//nn.setAllWeights(weights);
						
						double mse=0d;
						
						int testOffset=30;
						//Now test over the training set
						for(int i=0;i<30;i++){
							if(i==10) testOffset=80;
							if(i==20) testOffset=130;
							
							double [] res=nn.generateOutput(tests[testOffset+(i%10)].getParams());
							
							//System.out.println("Test:"+tests[testOffset+i]+" ; RES:"+res[0]+" "+res[1]+" "+res[2]);
							int indMax=0;
							double max=0d;
							for(int j=0;j<res.length;j++){
								if (j==0){max=res[j];indMax=1;}
								else {
									if(res[j]>max){
										max=res[j];
										indMax=j+1;
									}
								}
							}
							
							if(indMax!=tests[testOffset+(i%10)].getCorrectResult()){
								mse+=Math.pow(max-res[tests[testOffset+(i%10)].getCorrectResult()-1]+0.1,2);
							}
							
							
						}
						
						mse=mse/30;
						
						return mse;
		}
		
		public int correct;
		public double test(double [] weights){
			
			// Set chromosome's random generated weights to the network to test its perg on the whole training set
						//nn.setAllWeights(weights);
						correct=0;
						double mse=0d;
						
						int testOffset=40;
						//Now test over the training set
						for(int i=0;i<30;i++){
							if(i==10) testOffset=90;
							if(i==20) testOffset=140;
							
							double [] res=nn.generateOutput(tests[testOffset+(i%10)].getParams());
							
							//System.out.println("Test:"+tests[testOffset+i%10]+" ; RES:"+res[0]+" "+res[1]+" "+res[2]);
							int indMax=0;
							double max=0d;
							for(int j=0;j<res.length;j++){
								if (j==0){max=res[j];indMax=1;}
								else {
									if(res[j]>max){
										max=res[j];
										indMax=j+1;
									}
								}
							}
							
							
							if(indMax!=tests[testOffset+(i%10)].getCorrectResult()){
								mse+=Math.pow(max-res[tests[testOffset+(i%10)].getCorrectResult()-1]+0.1,2);
								System.out.println("DID NOT PASS indMax:"+indMax+" cooorrect:"+tests[testOffset+(i%10)].getCorrectResult()+"test:"+tests[testOffset+(i%10)].toString());
							}
							else {this.correct++;
							//System.out.println("PASSED");
							}
							
						}
						
						mse=mse/30;
						
						return mse;
		}
		
		
		public void iteration(){
			List<Chromosome> newGeneration=new ArrayList<Chromosome>();
			Chromosome bestPrev=new Chromosome(bestChromosome.getId(),bestChromosome.getFitness(),bestChromosome.getWeights());
			
			Random r = new Random();
			
			/*List<Chromosome> sample1= new ArrayList<Chromosome>();
			
			for(int i=0;i<this.sampleSizeSelection;i++){
				boolean b=sample1.add(this.population.get(r.nextInt(this.populationSize)));
				while(b==false){
					b=sample1.add(this.population.get(r.nextInt(this.populationSize)));
				}
			}
			
			Collections.sort(sample1);
			
			System.out.println(sample1.toString());
			
			for(int brBest=0;brBest<this.numOfBestToKeep;brBest++){
				newGeneration.add(sample1.get(brBest));
			}*/
			
			//Increment number of iterations
			this.numOfIterations++;
			
			double averageTMSE=0d;
			
			double averageVMSE=0d;
			
			while(newGeneration.size()!=this.populationSize){
				
				
				List<Chromosome> sample= new ArrayList<Chromosome>();
				
				for(int i=0;i<this.sampleSizeSelection;i++){
					boolean b=sample.add(this.population.get(r.nextInt(this.populationSize)));
					while(b==false){
						b=sample.add(this.population.get(r.nextInt(this.populationSize)));
					}
				}
				
				Chromosome p1=null;
				Chromosome p2=null;
				
				this.tournamentBestProb=0.8f;
				
				int r1=r.nextInt(this.sampleSizeSelection);
				int r2=r1;
				while(r2==r1){
				r2=r.nextInt(this.sampleSizeSelection);
				}
				int r3=r.nextInt(this.sampleSizeSelection);
				int r4=r3;
				while(r4==r3){
				r4=r.nextInt(this.sampleSizeSelection);
				}
				
				float prob=r.nextFloat();
				float prob1=r.nextFloat();
				
				if(sample.get(r2).getFitness()>=sample.get(r1).getFitness()){
					//r1 is better or equal
					if(prob<this.tournamentBestProb){
						p1=sample.get(r1);
					}
					else {p1=sample.get(r2);}
					
				}
				if(sample.get(r1).getFitness()>sample.get(r2).getFitness()){
					//r2 is better
					if(prob<this.tournamentBestProb){
						p1=sample.get(r2);
					}
					else {p1=sample.get(r1);}
					
				}
				if(sample.get(r4).getFitness()>=sample.get(r3).getFitness()){
					//r3 is better or equal
					if(prob1<this.tournamentBestProb){
						p2=sample.get(r3);
					}
					else {p2=sample.get(r4);}
					
				}
				if(sample.get(r3).getFitness()>sample.get(r4).getFitness()){
					//r4 is better
					if(prob1<this.tournamentBestProb){
						p2=sample.get(r4);
					}
					else {p2=sample.get(r3);}
					
				}
				
				//System.out.println("p1:"+p1);
				//System.out.println("p1:"+p2);
				
				//end of tournament, now we have parents, do crossover
				
				int worse=-1;
				if(p1.getFitness()<=p2.getFitness()) worse=1;
				else worse=0;
				
				
				float crossover1=r.nextFloat();
				
				if(crossover1<this.crossoverProb){
				//System.out.println("CROSSOVER HAPPENED");
				//num of numbers to copy to the worse parent
				int maxx=numWeights*3/5;
				//System.out.println("maxx:"+maxx);
				int minn=numWeights/3;
				//System.out.println("minn:"+minn);
				int crossover= r.nextInt(maxx-minn+1) +minn ;
				// from which index to start to copy from the better parent to the other
				int offsetStart=r.nextInt(this.numWeights);
				//System.out.println("crosssover:"+crossover+"offset st:"+offsetStart+"numwe"+numWeights);
				
				for(int k=0;k<crossover;k++){
					if(worse==0){
						p1.setWeight((offsetStart+k)%numWeights,p2.getWeights()[(offsetStart+k)%numWeights]);
						//System.out.println("worse p1"+p1);
					}
					else if(worse==1) {
						p2.setWeight((offsetStart+k)%numWeights,p1.getWeights()[(offsetStart+k)%numWeights]);
						//System.out.println("worse p2"+p1);
					}
				}
				
				
				// set new fitnesss to the worse
				if(worse==0){
					p1.setFitness(calculateFitness(p1.getWeights()));
				}
				else if(worse==1) {
					p2.setFitness(calculateFitness(p2.getWeights()));
				}
				
				
				} // end if crossover happens
				
				
				//System.out.println("AFTER CROSSOVER, p1="+p1+"\n and p2="+p2);
				// now mutatae
				
				float mutProb=r.nextFloat();
				
				if(mutProb<this.muatationProb){
					// Do mutation 
					
					//System.out.println("MUTATION OCCURS");
					
					int numOfMutations=(int)this.numWeights/5;
					
					int counter=0;
					
					while (counter<numOfMutations){
					counter++;
					int rand1=r.nextInt(this.numWeights);
					//System.out.printf("change "+rand1);
					
					double step=r.nextDouble() * (0.25d - (0.025d)) + (-0.025d);
					int leftOrRight=r.nextInt();
					
						if(worse==0){
							if(leftOrRight%2==0){
							p1.setWeight(rand1, p1.getWeight(rand1)+step);
							}
							else {
							p1.setWeight(rand1, p1.getWeight(rand1)-step);
							}
						}
						else {
							if(leftOrRight%2==0){
							p2.setWeight(rand1, p2.getWeight(rand1)+step);
							}
							else {
							p2.setWeight(rand1, p2.getWeight(rand1)-step);
							}
						}
					
					}
					
					// set new fitnesss to the worse
					if(worse==0){
						p1.setFitness(calculateFitness(p1.getWeights()));
					}
					else if(worse==1) {
						p2.setFitness(calculateFitness(p2.getWeights()));
					}
					
				
					//System.out.println("AFTER MUTATION p1:"+p1);
					//System.out.println("AFTER MUTATION p2:"+p2);
				} //end if mutation occurs
				
				
				if(worse==0){
					if(p1.getFitness()<bestChromosome.getFitness()){
						bestChromosome=new Chromosome(p1.getId(),p1.getFitness(),p1.getWeights());
					}
				}
				else if(worse==1){
					if(p2.getFitness()<bestChromosome.getFitness()){
						bestChromosome=new Chromosome(p2.getId(),p2.getFitness(),p2.getWeights());
					}
				}
				
				if(!newGeneration.contains(p1)){
				p1.setId(newGeneration.size());
				averageTMSE+=mseTrain;
				averageVMSE+=validate(p1.getWeights());
				newGeneration.add(p1);
				}
				
				if(newGeneration.size()==populationSize) break;
				
				//p2.setWeights(p1.getWeights());
				if(!newGeneration.contains(p2)){
				p2.setId(newGeneration.size());
				averageTMSE+=mseTrain;
				averageVMSE+=validate(p2.getWeights());
				newGeneration.add(p2);
				}
				//System.out.println(b+"size"+newGeneration.size()+"p1 eq p2:"+p1.equals(p2));
			
			}// keep choosing parents until a new generation is formed
			
			averageTMSE/=populationSize;
			averageVMSE/=populationSize;
			
			this.averageTrainingMSE.add(averageTMSE);
			this.averageValidationMSE.add(averageVMSE);
			
			//System.out.println("new gen"+newGeneration.toString());
			
		}
		
		
		public void trainNetwork(){
			
			int counter=0;
			int num=1100;
		
			//while(counter<400){
			while(true){
				counter++;
				if(this.averageTrainingMSE.get(counter-1)<0.00062d&&this.averageValidationMSE.get(counter-1)<0.0021d){
					break;
				}
				iteration();
				
				if(counter==10){
					nn=new NeuralNetwork(4);
					// create special init generation
					this.createInitialPopulation();
					System.out.println("NP:"+this.population.toString());
					
				}
			}
			
			System.out.println("Num of iterations:"+(counter-1));
			System.out.println("Best chromosome is:"+this.bestChromosome);
			System.out.println("Average training vector:"+this.averageTrainingMSE.toString());
			System.out.println("Average validation vector:"+this.averageValidationMSE.toString());
			
			int c1=0;
			int c2=0;
			for(int i=1;i<this.averageTrainingMSE.size();i++){
				if(this.averageTrainingMSE.get(i-1)<this.averageTrainingMSE.get(i)){
					c1++;
					System.out.print((i-1)+"&"+i+"; ");
				}
			}
			
			System.out.println("");
			
			for(int i=1;i<this.averageValidationMSE.size();i++){
				if(this.averageValidationMSE.get(i-1)<this.averageValidationMSE.get(i)){
					c2++;
					System.out.print((i-1)+"&"+i+"; ");
				}
			}
			
			System.out.println(" ");
			System.out.println("Num of inversions training:"+c1);
			System.out.println("Num of inversions validation:"+c2);
			
			double result=test(this.bestChromosome.getWeights());
			System.out.println("Test best chromosome found, result:"+result+" , correct out of 30: "+correct);
			
			
		}
		
		public void trainNetworkDynamic(int initNum){
			
			int counter=0;
			int num=1100;
			int numOfPerc=2;
			
			double prevAT;
			double prevVT;
			Chromosome lastBest;
			int offsetChange=0;
			int numNeurons=initNum;
			int thBias=0;
			
			List<Integer> changes=new ArrayList<Integer>();
			int threshold;
			
			int lastc=0;
			//while(counter<400){
			while(true){
				prevAT=this.averageTrainingMSE.get(counter);
				prevVT=this.averageValidationMSE.get(counter);
				lastBest=this.bestChromosome;
				
				counter++;
				if(this.averageTrainingMSE.get(counter-1)<0.00062d&&this.averageValidationMSE.get(counter-1)<0.0021d){
					break;
				}
				iteration();
				
				//if five invers consequtively incease number of neurons
				int c1=0;
				int lst1=offsetChange;
				int lst2=offsetChange+1;
				for(int i=offsetChange+1;i<this.averageTrainingMSE.size();i++){
					if(this.averageTrainingMSE.get(i-1)<this.averageTrainingMSE.get(i)||this.averageValidationMSE.get(i-1)<this.averageValidationMSE.get(i)){
					//	if(counter-1!=0&&lst1==(i-1)&&lst2==i){
						c1++;
					//	lst1=i-1;
					//	lst2=i;
						//System.out.println("FOUND CONSEQUTIVE "+c1);
					//	}
					//	else {
					//		c1=0;
					//	}
					}
				}
				
				threshold=(int)(this.averageTrainingMSE.size()-offsetChange)-(int)((this.averageTrainingMSE.size()-offsetChange)/(2+thBias));
				
				
				//System.out.println("CH:"+c1);
				
				if(threshold>4&&c1>=threshold){
					if(counter==0){
					lastc=c1;}
					else {
						if(c1<lastc&&numNeurons>2){
							numNeurons-=2;
						}
					}
					Random r=new Random();
					double dd=r.nextDouble();
					double p=0.85d;
					if(numNeurons%2==0 && dd<p){
					thBias+=1;
					}
					
					
					System.out.println("OVER THRESHOLD:"+threshold);
					offsetChange=counter;
					numNeurons++;
					
					List<Chromosome> ngen= new ArrayList<Chromosome>();
					
					System.out.println("NUM OF UNITS BEFORE:"+numNeurons+" nn:"+nn.numHiddenUnits);
					
					
					nn= new NeuralNetwork(numNeurons);
					
				
					// create special pop
					/*
					this.numWeights=nn.numInputs*nn.numHiddenUnits+nn.numHiddenUnits*nn.numOutputs;
					
					double averageTMSE=0d;
					double averageVMSE=0d;
				
					for(int bb=0;bb<this.populationSize;bb++){
						double [] ws=this.population.get(bb).getWeights();
						double[] ws1=new double[numWeights];
						System.out.println("WS:"+ws.length+" WS1:"+ws1.length+" NUMHIDDEN NEW"+numNeurons+" "+nn.numHiddenUnits);
						for(int bbb=0;bbb<ws.length-1;bbb++){
							ws1[bbb]=ws[bbb];
						}
						for(int bbb=ws.length-1;bbb<ws1.length;bbb++){
							ws1[bbb]=r.nextDouble() * (0.6d - (-0.6d)) + (-0.6d);
						}
						
						// Calculate fitness
						double fitness=calculateFitness(ws1);
						
						// Training set mse
						averageTMSE+=this.mseTrain;
						averageVMSE+=validate(ws1);
						Chromosome c=new Chromosome(bb, fitness,ws1);
						ngen.add(c);
						
						if(bb==0){
							bestChromosome= new Chromosome(c.getId(),c.getFitness(),c.getWeights());
						}
						else {
							if(c.getFitness()<bestChromosome.getFitness()){
								bestChromosome= new Chromosome(c.getId(),c.getFitness(),c.getWeights());
							}
						}
					}// end for all population
					
					averageTMSE/=populationSize;
					averageVMSE/=populationSize;
					
					this.averageTrainingMSE.add(averageTMSE);
					this.averageValidationMSE.add(averageVMSE);
					
					// end of creation of new generation out of old pop , assign it 
					this.population=ngen;*/
					
					createInitialPopulation();
					
					System.out.println("NP"+numNeurons+":"+this.population.toString());
				}
				
			}
			
			System.out.println("Num of neurons"+numNeurons);
			System.out.println("Num of iterations:"+(counter-1));
			System.out.println("Best chromosome is:"+this.bestChromosome);
			System.out.println("Average training vector wtf:"+this.averageTrainingMSE.toString());
			System.out.println("Average validation vector:"+this.averageValidationMSE.toString());
			
			double result=test(this.bestChromosome.getWeights());
			System.out.println("Test best chromosome found, result:"+result+" , correct out of 30: "+correct);
			
			
		}
		
	

}
