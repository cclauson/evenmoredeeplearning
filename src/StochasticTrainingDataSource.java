import java.util.ArrayList;
import java.util.List;

public class StochasticTrainingDataSource implements TrainingDataSource {
	
	private final List<XValsYValsPair> pairs;
	private int index = 0;
	
	public StochasticTrainingDataSource(XValsYValsPair allData) {
		this.pairs = new ArrayList<XValsYValsPair>();
		for (int i = 0; i < allData.numCases(); ++i) {
			Matrix xVector = new Matrix(1, allData.inputDim());
			Matrix yVector = new Matrix(1, allData.outputDim());
			for (int j = 0; j < allData.inputDim(); ++j) {
				float val = allData.getTransposedXVals().get(j, i);
				xVector.set(0, j, val);
			}
			for (int j = 0; j < allData.outputDim(); ++j) {
				float val = allData.getTransposedYVals().get(j, i);
				yVector.set(0, j, val);
			}
			pairs.add(new XValsYValsPair(xVector, yVector));
		}
	}
	
	@Override
	public XValsYValsPair getTrainingPair() {
		XValsYValsPair ret = pairs.get(index);
		++index;
		index %= pairs.size();
		return ret;
	}

}
