import java.util.function.Function;

public class FullyConnectedLayer {
	
	private final int inputDim;
	private final int outputDim;
	
	//private Matrix params;
	private final AbstractParamMatrix abstractParamMatrix;
	
	private Matrix lastU;
	private Matrix lastX;
	
	private final Sigmoid sigmoid;

	public Matrix paramsMatrix() {
		return abstractParamMatrix.getMatrix();
	}
	
	public FullyConnectedLayer(Matrix matrix, Sigmoid sigmoid) {
		this.inputDim = matrix.getN() - 1;
		this.outputDim = matrix.getM();
		this.sigmoid = sigmoid;
		this.abstractParamMatrix = new MatrixBasedAbstractMatrix(matrix);
	}
	
	public FullyConnectedLayer(int inputDim, int outputDim, Sigmoid sigmoid) {
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		this.sigmoid = sigmoid;
		/*
		this.params = new Matrix(outputDim, inputDim + 1);
		final Random rand = new Random();
		for (int i = 0; i < params.getM(); ++i) {
			for (int j = 0; j < params.getN(); ++j) {
				params.set(i, j, 0.45f + rand.nextFloat() * 0.1f - 0.5f);
			}
		}
		*/
		this.abstractParamMatrix = new MatrixBasedAbstractMatrix(outputDim, inputDim + 1);
		//this.abstractParamMatrix = new VectorMappingParamMatrix(outputDim, inputDim + 1);
		//this.abstractParamMatrix = new SimpleSaturatingMatrix(outputDim, inputDim + 1);
	}
	
	private static Matrix augmentBottomWithOnes(Matrix matrix) {
		final Matrix ret = new Matrix(matrix.getM() + 1, matrix.getN());
		for (int i = 0; i < ret.getM(); ++i) {
			for (int j = 0; j < ret.getN(); ++j) {
				final float val = (i == matrix.getM())? 1.0f : matrix.get(i, j);
				ret.set(i, j, val);
			}
		}
		return ret;
	}

	private static Matrix stripBottomRow(Matrix matrix) {
		final Matrix ret = new Matrix(matrix.getM() - 1, matrix.getN());
		for (int i = 0; i < ret.getM(); ++i) {
			for (int j = 0; j < ret.getN(); ++j) {
				final float val = matrix.get(i, j);
				ret.set(i, j, val);
			}
		}
		return ret;
	}
	
	public Matrix evalForward(Matrix inputMatrix) {
		if (inputMatrix.getM() != this.inputDim) {
			throw new IllegalArgumentException("inputMatrixDim (" +
					inputMatrix.getM() + ") not expected dim (" +
					this.inputDim + ")");
		}
		
		//for multilayer networks we can't require this
		/*
		//verify that every matrix entry is either 0.0f or 1.0f
		for (int i = 0; i < inputMatrix.getM(); ++i) {
			for (int j = 0; j < inputMatrix.getN(); ++j) {
				if (inputMatrix.get(i, j) != 0.0f && inputMatrix.get(i, j) != 1.0f) {
					throw new IllegalArgumentException("Found non-0.0f/1.0f entry");
				}
			}
		}
		*/
		inputMatrix = augmentBottomWithOnes(inputMatrix);
		
		System.out.println("Input matrix X:");
		System.out.println(inputMatrix);
		System.out.println();

		System.out.println("Param matrix W:");
		System.out.println(this.abstractParamMatrix.getMatrix());
		System.out.println();

		final Matrix u = this.abstractParamMatrix.getMatrix().mul(inputMatrix);
		System.out.println("Matrix U = W*X:");
		System.out.println(u);
		System.out.println();

		//cache U and X for back propagation
		this.lastU = u;
		this.lastX = inputMatrix;
		
		final Matrix y = new Matrix(u);
		y.applyElementwiseFunctionInPlace(new Function<Float, Float>() {
			@Override public Float apply(Float val) { return sigmoid.eval(val); }
		});
		
		System.out.println("matrix Y = sigmoid(U):");
		System.out.println(y);
		System.out.println();
		
		return y;
		
	}
	
	public Matrix backPropagate(Matrix partialEPartialY, float eta) {
		if (this.lastU == null) {
			throw new IllegalStateException("no last u");
		}
		if (this.lastX == null) {
			throw new IllegalStateException("no last x");
		}
		if (this.lastU.getM() != this.outputDim) {
			throw new IllegalStateException("last u vertical dimension not equal to output dim");
		}
		if (partialEPartialY.getM() != this.outputDim) {
			throw new IllegalArgumentException("partial e partial y vertical dimension not equal to output dim");
		}
		if (partialEPartialY.getN() != this.lastU.getN()) {
			throw new IllegalArgumentException("partial e partial y has different horiz. dimension than last u " +
					"(i.e., different number of training pairs)");
		}
		
		this.lastU.applyElementwiseFunctionInPlace(new Function<Float, Float>() {
			@Override public Float apply(Float val) { return sigmoid.evalDerivative(val); }
		});
		
		final Matrix partialEPartialU = this.lastU.mulHadamard(partialEPartialY);
		System.out.println("partial E/partial U = sigmoid'(U) o (sigmoid(U) - Y_train)");
		System.out.println(partialEPartialU);
		System.out.println();

		//compute the sensitivity of the error with respect to inputs
		Matrix partialEPartialX = this.abstractParamMatrix.getMatrix().transpose().mul(partialEPartialU);
		partialEPartialX = stripBottomRow(partialEPartialX);
		System.out.println("partial E/partial X");
		System.out.println(partialEPartialX);
		System.out.println();
		
		
		final Matrix partialEPartialW = partialEPartialU.mul(this.lastX.transpose());
		System.out.println("partial E/partial W");
		System.out.println(partialEPartialW);
		System.out.println();

		System.out.println("Old param matrix W:");
		System.out.println(this.abstractParamMatrix.getMatrix());
		System.out.println();
		
		/*
		partialEPartialW.scaleBy(0.08f);
		this.params = params.sub(partialEPartialW);
		*/
		
		this.abstractParamMatrix.updateWithDeltas(partialEPartialW, eta);
		this.lastU = null;
		this.lastX = null;
		
		System.out.println("New param matrix W:");
		System.out.println(this.abstractParamMatrix.getMatrix());
		System.out.println();
		
		return partialEPartialX;
		
	}
	
}
