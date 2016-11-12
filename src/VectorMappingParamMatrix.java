import java.util.*;

//represents a parameter matrix which is the image of a different
//matrix, where each row of the parameter matrix is the image
//of a (possibly nonlinear) function of the corresponding row
//of the underlying matrix
public class VectorMappingParamMatrix implements AbstractParamMatrix {

	//each VectorPreimage object is capable
	//of producing a row of the output matrix
	private List<VectorPreimage> vectorPreimages;
	
	private final int m;
	private final int n;
	
	public VectorMappingParamMatrix(int m, int n) {
		this.m = m;
		this.n = n;
		vectorPreimages = new ArrayList<VectorPreimage>();
		for (int i = 0; i < m; ++i) {
			vectorPreimages.add(new SimpleVectorPreimage(n));
		}
	}
	
	@Override
	public Matrix getMatrix() {
		Matrix ret = new Matrix(m, n);
		for (int i = 0; i < m; ++i) {
			final List<Float> vectorImage = vectorPreimages.get(i).getVectorImage();
			for (int j = 0; j < n; ++j) {
				ret.set(i, j, vectorImage.get(j));
			}
		}
		return ret;
	}
	
	private List<Float> getMatrixRowAsFloatList(Matrix matrix, int i) {
		final List<Float> ret = new ArrayList<Float>();
		for (int j = 0; j < matrix.getN(); ++j) {
			ret.add(matrix.get(i, j));
		}
		return ret;
	}
	
	@Override
	public void updateWithDeltas(Matrix deltasMatrix, float eta) {
		if (deltasMatrix.getM() != this.m) {
			throw new IllegalArgumentException("invalid matrix m");
		}
		if (deltasMatrix.getN() != this.n) {
			throw new IllegalArgumentException("invalid matrix n");
		}
		for (int i = 0; i < m; ++i) {
			final List<Float> row = getMatrixRowAsFloatList(deltasMatrix, i);
			this.vectorPreimages.get(i).backPropagate(row, eta);
		}
	}

}
