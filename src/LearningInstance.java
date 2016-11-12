
public class LearningInstance {

	public final int inputDim;
	public final int middleDim;
	public final int outputDim;
	
	public final XValsYValsPair xValsYValsPair;
	
	public LearningInstance(
		int inputDim,
		int middleDim,
		int outputDim,
		XValsYValsPair xValsYValsPair
	) {
		this.inputDim = inputDim;
		this.middleDim = middleDim;
		this.outputDim = outputDim;
		if (xValsYValsPair.inputDim() != inputDim) {
			throw new IllegalArgumentException("input matrix has wrong horizontal dimension");
		}
		if (xValsYValsPair.outputDim() != outputDim) {
			throw new IllegalArgumentException("output matrix has wrong horizontal dimension");
		}
		this.xValsYValsPair = xValsYValsPair;
	}
	
}
