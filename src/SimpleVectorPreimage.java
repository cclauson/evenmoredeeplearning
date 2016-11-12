import java.util.*;

public class SimpleVectorPreimage implements VectorPreimage {

	private final List<Float> params;
	private final int outputDim;
	
	private final Matrix forwardMatrix;
	private final Matrix forwardMatrixTranspose;
	
	private static Matrix buildForwardMatrix(int outputDim) {
		final Matrix ret = new Matrix(outputDim, 2 * outputDim);
		for (int i = 0; i < outputDim; ++i) {
			ret.set(i, i * 2, 1.0f);
			ret.set(i, i * 2 + 1, -1.0f);
			ret.set(outputDim - 1, i * 2 + 1, ret.get(outputDim - 1, i * 2 + 1) + 1.0f);
		}
		//actually, we want to map the output from the range
		//[0.0, 1.0] to the range [-1.0, 1.0], so multiply
		//each entry by 2, and subtract 1 from the last column
		/*
		for (int i = 0; i < outputDim; ++i) {
			for (int j = 0; j < outputDim; ++j) {
				ret.set(i, j, 2 * ret.get(i, j));
			}
			//ret.set(i, ret.getN() - 1, ret.get(i, ret.getN() - 1) - 1.0f);
		}
		//*/
		return ret;
	}
	
	//impose saturation so that param is on [0.0, 1.0],
	//then divide params by sum so that vector sums
	//to 1.0
	private void normalizeParams() {
		float sum = 0.0f;
		for (int i = 0; i < params.size(); ++i) {
			float val = params.get(i);
			params.set(i, val);
			sum += val;
		}
		for (int i = 0; i < params.size(); ++i) {
			params.set(i, params.get(i)/sum);
		}
	}
	
	public SimpleVectorPreimage(int outputDim) {
		this.params = new ArrayList<Float>();
		final Random rand = new Random();
		while (this.params.size() < outputDim * 2) {
			params.add(0.45f + 0.1f * rand.nextFloat());
		}
		normalizeParams();
		this.outputDim = outputDim;
		this.forwardMatrix = buildForwardMatrix(outputDim);
		//System.out.println("Forward matrix: ");
		//System.out.println(forwardMatrix);
		this.forwardMatrixTranspose  = this.forwardMatrix.transpose();
		//System.out.println("Forward matrix transpose: ");
		//System.out.println(forwardMatrixTranspose);
	}

	private static List<Float> mulMatrixByListFloat(Matrix matrix, List<Float> inputList) {
		final Matrix colVectorIn = new Matrix(inputList);
		final Matrix colVectorOut = matrix.mul(colVectorIn);
		return colVectorOut.getColumnVectorAsFloatList();
	}
	
	@Override
	public List<Float> getVectorImage() {
		//System.out.println(params);		
		return mulMatrixByListFloat(this.forwardMatrix, this.params);
		/*
		final List<Float> ret = new ArrayList<Float>();
		for (int i = 0; i < params.size(); i += 2) {
			ret.add(params.get(i) - params.get(i + 1));
		}
		for (int i = 0; i < params.size(); i += 2) {
			ret.set(ret.size() - 1, ret.get(ret.size() - 1) + params.get(i + 1));
		}
		return ret;
		*/
	}
	
	@Override
	public void backPropagate(List<Float> partialEPartialOutput, float eta) {

		System.out.println("Initial params (W):");
		System.out.println(getVectorImage());
		
		//compute sensitivity of outputs with respect to params
		System.out.println("Param (W) changes:");
		System.out.println(partialEPartialOutput);
		System.out.println("Matrix:");
		System.out.println(this.forwardMatrixTranspose);
		final List<Float> paramChanges =
				mulMatrixByListFloat(this.forwardMatrixTranspose, partialEPartialOutput);
		System.out.println("Param (V) changes:");
		System.out.println(paramChanges);
		if (paramChanges.size() != this.params.size()) {
			throw new RuntimeException("param changes dimension unexpectedly not equal " +
					"to param dimension");
		}
		System.out.println("Current (V) params:");
		System.out.println(this.params);
		//now update params
		for (int i = 0; i < this.params.size(); ++i) {
			float newVal = params.get(i) - eta * paramChanges.get(i);
			params.set(i, newVal);
		}
		normalizeParams();
		System.out.println("New (V) params:");
		System.out.println(this.params);

		System.out.println("New params (W):");
		System.out.println(getVectorImage());
		System.out.println();
		
		/*
		if (partialEPartialOutput.size() != this.outputDim) {
			throw new IllegalArgumentException("bad input vector dimension");
		}
		final List<Float> deltaParams = new ArrayList<Float>();
		while (deltaParams.size() < params.size()) {
			deltaParams.add(0.0f);
		}
		for (int i = 0; i < params.size(); i += 2) {
			float val = partialEPartialOutput.get(i/2);
			deltaParams.set(i, val);
			deltaParams.set(i + 1, -val);
		}
		*/
	}

}
