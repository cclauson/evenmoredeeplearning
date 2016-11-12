import java.util.Random;

public class Main {

	private static float high = 1.0f;
	//private static float low = 0.0f;
	private static float low = -1.0f;
	
	private Main() {}
	
	private static void runLearningInstance(
		LearningInstance learningInstance,
		TrainingDataSource trainingDataSource,
		int numIterations
	) {
		
		float eta = 0.01f;
		
		//final Sigmoid sigmoid = Sigmoid.ONE_HALF_CENTERED_TO_ONE_HALF_CENTERED;
		//final Sigmoid sigmoid = Sigmoid.ONE_HALF_CENTERED;
		final Sigmoid sigmoid = Sigmoid.ZERO_CENTERED_TO_ZERO_CENTERED;
		
		FullyConnectedLayer fcl1 = new FullyConnectedLayer(
				learningInstance.inputDim, learningInstance.middleDim, sigmoid);
		FullyConnectedLayer fcl2 = new FullyConnectedLayer(
				learningInstance.middleDim, learningInstance.outputDim, sigmoid);
		
		for (int i = 0; i < numIterations; ++i) {

			XValsYValsPair trainingDataForIteration = trainingDataSource.getTrainingPair();
			
			Matrix y_mid = fcl1.evalForward(trainingDataForIteration.getTransposedXVals());
			Matrix y = fcl2.evalForward(y_mid);
			
			System.out.println("Y_train:");
			System.out.println(trainingDataForIteration.getTransposedYVals());
			System.out.println();

			Matrix partialEPartialY = y.sub(trainingDataForIteration.getTransposedYVals());
			partialEPartialY.scaleBy(2.0f);

			System.out.println("partial-E / partial-Y = Y - Y-train:");
			System.out.println(partialEPartialY);
			System.out.println();

			Matrix partialEPartialX_mid = fcl2.backPropagate(partialEPartialY, eta);
			Matrix partialEPartialX = fcl1.backPropagate(partialEPartialX_mid, eta);
			
		}

		System.out.println("Checking forward eval 1");
		System.out.println();
		checkForwardEval(fcl1, fcl2, learningInstance.xValsYValsPair);
		
		fcl1 = new FullyConnectedLayer(
				normalizeParamsMatrix(fcl1.paramsMatrix()), sigmoid);
		fcl2 = new FullyConnectedLayer(
				normalizeParamsMatrix(fcl2.paramsMatrix()), sigmoid);

		System.out.println("Checking forward eval 2");
		System.out.println();
		checkForwardEval(fcl1, fcl2, learningInstance.xValsYValsPair);

	}
	
	private static Matrix normalizeParamsMatrix(Matrix inputMatrix) {
		for (int i = 0; i < inputMatrix.getM(); ++i) {
			float max = 0.0f;
			for (int j = 0; j < inputMatrix.getN(); ++j) {
				float valAbs = Math.abs(inputMatrix.get(i, j));
				if (valAbs > max) max = valAbs;
			}
			if (max == 0.0f) throw new RuntimeException(
					"zero row not expected");
			for (int j = 0; j < inputMatrix.getN(); ++j) {
				inputMatrix.set(i, j, inputMatrix.get(i, j) /
						(max * inputMatrix.getN()));
			}
		}
		return inputMatrix;
	}
	
	private static void checkForwardEval(FullyConnectedLayer fcl1,
			FullyConnectedLayer fcl2, XValsYValsPair xValsYValsPair) {
		Matrix y_mid = fcl1.evalForward(
				xValsYValsPair.getTransposedXVals());
		Matrix y = fcl2.evalForward(y_mid);

		System.out.println("Y final:");
		System.out.println(y.transpose());
		System.out.println();
		
		System.out.println("Y training:");
		System.out.println(
				xValsYValsPair.getTransposedYVals().transpose());
		System.out.println();
	}
	
	private static LearningInstance makeStandardLearningInstance() {
		final Matrix xVals = new Matrix(8, 8);
		for (int i = 0; i < 8; ++i) {
			for (int j = 0; j < 8; ++j) {
				float val = i == j? high : low;
				xVals.set(i, j, val);
			}
		}
		final Matrix yVals = new Matrix(xVals);
		return new LearningInstance(8, 3, 8,
					new XValsYValsPair(xVals, yVals));
	}
	
	private static LearningInstance makeRandomLearningInstance() {
		int middleLayerSize = 4;
		Random rand = new Random();
		final Matrix xVals = new Matrix(8, 8);
		final Matrix yVals = new Matrix(8, 8);
		for (int i = 0; i < 8; ++i) {
			for (int j = 0; j < 8; ++j) {
				float val = (rand.nextFloat() < 0.3f)? high : low;
				xVals.set(i, j, val);
				val = (rand.nextFloat() < 0.3f)? high : low;
				yVals.set(i, j, val);
			}
		}
		return new LearningInstance(8, middleLayerSize, 8,
					new XValsYValsPair(xVals, yVals));
	}
	
	private static void runARandomLearningInstance() {
		
		//final LearningInstance learningInstance =
		//		makeRandomLearningInstance();
		final LearningInstance learningInstance =
				makeStandardLearningInstance();
		
		TrainingDataSource tds;
		if (true) {
			tds = new TrainingDataSource() {
				@Override
				public XValsYValsPair getTrainingPair() {
					return learningInstance.xValsYValsPair;
				}
			};
		} else {
			tds = new StochasticTrainingDataSource(learningInstance.xValsYValsPair);
		}
		int numIterations = 10000;
		//int numIterations = 1;
		runLearningInstance(learningInstance, tds, numIterations);
	}
	
	public static void main(String[] args) {
		runARandomLearningInstance();
	}
	
}
