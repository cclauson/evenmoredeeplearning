import java.util.Random;

//simplest implementation, backed by a parameter
//matrix, update by adding in eta times the deltas
public class MatrixBasedAbstractMatrix implements AbstractParamMatrix {

	private Matrix paramsMatrix;
	
	public MatrixBasedAbstractMatrix(int m, int n) {
		this.paramsMatrix = new Matrix(m, n);
		final Random rand = new Random();
		for (int i = 0; i < paramsMatrix.getM(); ++i) {
			for (int j = 0; j < paramsMatrix.getN(); ++j) {
				paramsMatrix.set(i, j, 0.45f + rand.nextFloat() * 0.1f - 0.5f);
			}
		}
	}
	
	public MatrixBasedAbstractMatrix(Matrix matrix) {
		this.paramsMatrix = matrix;
	}
	
	@Override
	public Matrix getMatrix() {
		return this.paramsMatrix;
	}
	
	@Override
	public void updateWithDeltas(Matrix deltasMatrix, float eta) {
		final Matrix deltasMatrixCopy = new Matrix(deltasMatrix);
		deltasMatrixCopy.scaleBy(eta);
		this.paramsMatrix = paramsMatrix.sub(deltasMatrixCopy);
	}

}
