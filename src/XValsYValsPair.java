
public class XValsYValsPair {
	
	private final Matrix xVals;
	private final Matrix yVals;
	private final Matrix xValsTranspose;
	private final Matrix yValsTranspose;
	
	public XValsYValsPair(Matrix xVals, Matrix yVals) {
		this.xVals = new Matrix(xVals);
		this.yVals = new Matrix(yVals);
		if (xVals.getM() != yVals.getM()) {
			throw new IllegalArgumentException(
					"x and y matrices do not have the same number of cases");
		}
		this.xValsTranspose = xVals.transpose();
		this.yValsTranspose = yVals.transpose();
	}
	
	public int inputDim() { return xVals.getN(); }
	public int outputDim() { return yVals.getN(); }
	public int numCases() { return xVals.getM(); }
	
	public Matrix getTransposedXVals() { return xValsTranspose; }
	public Matrix getTransposedYVals() { return yValsTranspose; }
	
}
