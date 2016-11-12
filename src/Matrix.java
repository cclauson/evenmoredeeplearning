import java.text.DecimalFormat;
import java.util.*;
import java.util.function.Function;

public class Matrix {

	private final float[] data;
	private final int m;
	private final int n;
	
	public Matrix(Matrix matrix) {
		this.m = matrix.m;
		this.n = matrix.n;
		this.data = Arrays.copyOf(matrix.data, matrix.data.length);
	}
	
	public Matrix(int m, int n) {
		this.m = m;
		this.n = n;
		this.data = new float[m * n];
	}
	
	//construct a column vector
	public Matrix(List<Float> data) {
		this(data.size(), 1);
		for (int i = 0; i < data.size(); ++i) {
			this.data[i] = data.get(i);
		}
	}
	
	public boolean isColumnVector() {
		return n == 1;
	}
	
	public List<Float> getColumnVectorAsFloatList() {
		if (!this.isColumnVector())
			throw new IllegalArgumentException("only valid for column vector");
		final List<Float> ret = new ArrayList<Float>();
		for (int i = 0; i < this.data.length; ++i) {
			ret.add(this.data[i]);
		}
		return ret;
	}
	
	public int getM() { return this.m; }
	public int getN() { return this.n; }
	
	private int boundCheckIndicesAndGetAbsoluteIndex(
			int i, int j) {
		if (i < 0) throw new IllegalArgumentException("i is negative");
		if (i >= m) throw new IllegalArgumentException("i is >= m");
		if (j < 0) throw new IllegalArgumentException("j is negative");
		if (j >= n) throw new IllegalArgumentException("j is >= n");
		return j * m + i;
	}
	
	public void set(int i, int j, float val) {
		int index = boundCheckIndicesAndGetAbsoluteIndex(i, j);
		data[index] = val;
	}
	
	public float get(int i, int j) {
		int index = boundCheckIndicesAndGetAbsoluteIndex(i, j);
		return data[index];
	}
	
	public Matrix mul(Matrix other) {
		if (this.n != other.m)
			throw new IllegalArgumentException("matrix dimensions not compatible (" +
					m + " x " + n + ") times (" + other.m + " x " + other.n + ")");
		final Matrix ret = new Matrix(m, other.n);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < other.n; ++j) {
				float sum = 0.0f;
				for (int k = 0; k < n; ++k) {
					sum += get(i, k) * other.get(k, j);
				}
				ret.set(i, j, sum);
			}
		}
		return ret;
	}
	
	public Matrix transpose() {
		Matrix ret = new Matrix(n, m);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				ret.set(j, i, get(i, j));
			}
		}
		return ret;
	}
	
	public void applyElementwiseFunctionInPlace(Function<Float, Float> func) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				int index = boundCheckIndicesAndGetAbsoluteIndex(i, j);
				data[index] = func.apply(data[index]);
			}
		}
	}
	
	private void checkMatrixDimEqual(Matrix other) {
		if (other.m != this.m) throw new IllegalArgumentException("m dimensions not equal"); 
		if (other.n != this.n) throw new IllegalArgumentException("n dimensions not equal"); 
	}
	
	public Matrix sub(Matrix other) {
		checkMatrixDimEqual(other);
		final Matrix ret = new Matrix(this);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				int index = boundCheckIndicesAndGetAbsoluteIndex(i, j);
				ret.data[index] = ret.data[index] - other.data[index];
			}
		}
		return ret;
	}
	
	public Matrix mulHadamard(Matrix other) {
		checkMatrixDimEqual(other);
		final Matrix ret = new Matrix(this);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				int index = boundCheckIndicesAndGetAbsoluteIndex(i, j);
				ret.data[index] = ret.data[index] * other.data[index];
			}
		}
		return ret;
	}
	
	public void scaleBy(float scalar) {
		this.applyElementwiseFunctionInPlace(new Function<Float, Float>() {
			@Override
			public Float apply(Float val) {
				return val * scalar;
			}
		});
	}
		
	@Override
	public String toString() {
		final StringBuilder sb = new StringBuilder();
		final DecimalFormat df = new DecimalFormat(" 0.000;-0.000");
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				sb.append(df.format(get(i, j)) + "  ");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
	
}
