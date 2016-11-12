import java.util.List;

public interface VectorPreimage {
	public List<Float> getVectorImage();
	public void backPropagate(List<Float> partialEPartialOutput, float eta);
}
