package nn;

import java.util.Arrays;

public class Test {

	private double [] params;
	
	private int correctResult;
	
	public Test(double [] params, int correctResult){
		this.correctResult=correctResult;
		this.params=new double [4];
		this.params[0]=params[0];
		this.params[1]=params[1];
		this.params[2]=params[2];
		this.params[3]=params[3];
	}
	
	public double getParam(int i){
		return this.params[i];
	}
	
	public void setParam(int i, double v){
		this.params[i]=v;
	}

	public double[] getParams() {
		return params;
	}

	public void setParams(double[] params) {
		this.params = params;
	}

	public int getCorrectResult() {
		return correctResult;
	}

	public void setCorrectResult(int correctResult) {
		this.correctResult = correctResult;
	}

	@Override
	public String toString() {
		return "Test [params=" + Arrays.toString(params) + ", correctResult="
				+ correctResult + "]";
	}
	
	
}
