import java.util.Random;
import java.util.function.Function;

public class SimpleSaturatingMatrix implements AbstractParamMatrix {

	private Matrix paramsMatrix;
	
	public SimpleSaturatingMatrix(int m, int n) {
		this.paramsMatrix = new Matrix(m, n);
		final Random rand = new Random();
		for (int i = 0; i < paramsMatrix.getM(); ++i) {
			for (int j = 0; j < paramsMatrix.getN(); ++j) {
				paramsMatrix.set(i, j, rand.nextFloat() * 0.1f - 0.05f);
			}
		}
	}
	
	@Override
	public Matrix getMatrix() {
		final Matrix ret = new Matrix(this.paramsMatrix);
		ret.scaleBy(1.0f/paramsMatrix.getN());
		return ret;
	}
	
	@Override
	public void updateWithDeltas(Matrix deltasMatrix, float eta) {
		final Matrix deltasMatrixCopy = new Matrix(deltasMatrix);
		deltasMatrixCopy.scaleBy(eta * paramsMatrix.getN());
		this.paramsMatrix = paramsMatrix.sub(deltasMatrixCopy);
		for (int i = 0; i < paramsMatrix.getM(); ++i) {
			for (int j = 0; j < paramsMatrix.getM(); ++j) {
				this.paramsMatrix.applyElementwiseFunctionInPlace(new Function<Float, Float>() {
					@Override
					public Float apply(Float arg0) {
						if (arg0 > 1.0f) return 1.0f;
						if (arg0 < -1.0f) return -1.0f;
						return arg0;
					}
				});
			}
		}
	}

}
