
public interface AbstractParamMatrix {
	public Matrix getMatrix();
	public void updateWithDeltas(Matrix deltasMatrix, float eta);
}
