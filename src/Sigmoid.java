
public interface Sigmoid {

	public float eval(float val);
	public float evalDerivative(float val);
	
	public static final Sigmoid ZERO_CENTERED_TO_ONE_HALF_CENTERED = new Sigmoid() {
		@Override
		public float eval(float val) {
			float tanh = (float)Math.tanh(val);
			return (tanh + 1.0f) / 2;
		}
		@Override
		public float evalDerivative(float val) {
			float tanh = (float)Math.tanh(val);
			return (1 - tanh * tanh) / 2;
		}
	};

	public static final Sigmoid ONE_HALF_CENTERED_TO_ONE_HALF_CENTERED  = new Sigmoid() {
		
		private float bias(float val) {
			return val * 2.0f - 1.0f;
		}
		
		@Override
		public float eval(float val) {
			return ZERO_CENTERED_TO_ONE_HALF_CENTERED.eval(bias(val));
		}
		@Override
		public float evalDerivative(float val) {
			return ZERO_CENTERED_TO_ONE_HALF_CENTERED.evalDerivative(bias(val));
		}
	};

	public static final Sigmoid ZERO_CENTERED_TO_ZERO_CENTERED = new Sigmoid() {
		@Override
		public float eval(float val) {
			float tanh = (float)Math.tanh(val);
			return tanh;
		}
		@Override
		public float evalDerivative(float val) {
			float tanh = (float)Math.tanh(val);
			return 1 - tanh * tanh;
		}
	};

	public static final Sigmoid ONE_HALF_CENTERED_TO_ZERO_CENTERED  = new Sigmoid() {
		
		private float bias(float val) {
			return val * 2.0f - 1.0f;
		}
		
		@Override
		public float eval(float val) {
			return ZERO_CENTERED_TO_ZERO_CENTERED.eval(bias(val));
		}
		@Override
		public float evalDerivative(float val) {
			return ZERO_CENTERED_TO_ZERO_CENTERED.evalDerivative(bias(val));
		}
	};

}
