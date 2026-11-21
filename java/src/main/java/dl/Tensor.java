package dl;

import java.util.Arrays;
import java.util.Random;

/**
 * A minimal N-dimensional tensor (row-major, like NumPy/PyTorch).
 *
 * data    : flat array holding every element
 * shape   : size of each dimension, e.g. {2,3} = 2 rows, 3 columns
 * strides : how many flat positions you jump when an index increases by 1
 *           for shape {2,3}: strides = {3,1}  ->  element (i,j) lives at i*3 + j*1
 */
public class Tensor {
    public final double[] data;
    public final int[] shape;
    public final int[] strides;

    public Tensor(double[] data, int... shape) {
        if (data.length != numel(shape))
            throw new IllegalArgumentException("data length " + data.length + " != prod(shape) " + numel(shape));
        this.data = data;
        this.shape = shape.clone();
        this.strides = computeStrides(shape);
    }

    // ------------------------------------------------------------ factories
    public static Tensor zeros(int... shape) { return new Tensor(new double[numel(shape)], shape); }

    public static Tensor full(double v, int... shape) {
        double[] d = new double[numel(shape)];
        Arrays.fill(d, v);
        return new Tensor(d, shape);
    }

    public static Tensor randn(Random rng, double std, int... shape) {
        double[] d = new double[numel(shape)];
        for (int i = 0; i < d.length; i++) d[i] = rng.nextGaussian() * std;
        return new Tensor(d, shape);
    }

    public static Tensor arange(int n) {
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = i;
        return new Tensor(d, n);
    }

    public static Tensor of2d(double[][] rows) {
        int r = rows.length, c = rows[0].length;
        double[] d = new double[r * c];
        for (int i = 0; i < r; i++) System.arraycopy(rows[i], 0, d, i * c, c);
        return new Tensor(d, r, c);
    }

    // ------------------------------------------------------------ helpers
    public static int numel(int[] shape) {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    static int[] computeStrides(int[] shape) {
        int[] st = new int[shape.length];
        int acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) { st[i] = acc; acc *= shape[i]; }
        return st;
    }

    public int ndim() { return shape.length; }
    public int size() { return data.length; }

    public double get(int... idx) { return data[offset(idx)]; }
    public void set(double v, int... idx) { data[offset(idx)] = v; }

    int offset(int[] idx) {
        if (idx.length != shape.length) throw new IllegalArgumentException("wrong number of indices");
        int o = 0;
        for (int i = 0; i < idx.length; i++) {
            if (idx[i] < 0 || idx[i] >= shape[i]) throw new IndexOutOfBoundsException(Arrays.toString(idx));
            o += idx[i] * strides[i];
        }
        return o;
    }

    public Tensor copy() { return new Tensor(data.clone(), shape); }

    // ------------------------------------------------------------ reshape
    /** Same data, new shape. One dimension may be -1 (inferred). No copy: only the "view" changes. */
    public Tensor reshape(int... newShape) {
        int[] s = newShape.clone();
        int infer = -1, known = 1;
        for (int i = 0; i < s.length; i++) {
            if (s[i] == -1) { if (infer >= 0) throw new IllegalArgumentException("only one -1"); infer = i; }
            else known *= s[i];
        }
        if (infer >= 0) s[infer] = data.length / known;
        if (numel(s) != data.length)
            throw new IllegalArgumentException("cannot reshape " + Arrays.toString(shape) + " to " + Arrays.toString(newShape));
        return new Tensor(data, s);
    }

    /** 2-D transpose (copies data). */
    public Tensor transpose() {
        check2d();
        int r = shape[0], c = shape[1];
        Tensor t = zeros(c, r);
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) t.data[j * r + i] = data[i * c + j];
        return t;
    }

    // ------------------------------------------------------------ broadcasting
    /**
     * Broadcasting rule (same as NumPy): align shapes from the RIGHT;
     * two dimensions are compatible if equal or one of them is 1.
     * Example: (3,1) op (1,4) -> (3,4);  (2,3) op (3,) -> (2,3);  (2,3) op (2,) -> ERROR
     */
    public static int[] broadcastShape(int[] a, int[] b) {
        int n = Math.max(a.length, b.length);
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int da = i < n - a.length ? 1 : a[i - (n - a.length)];
            int db = i < n - b.length ? 1 : b[i - (n - b.length)];
            if (da != db && da != 1 && db != 1)
                throw new IllegalArgumentException("shapes " + Arrays.toString(a) + " and " + Arrays.toString(b) + " not broadcastable");
            out[i] = Math.max(da, db);
        }
        return out;
    }

    public interface Op { double apply(double x, double y); }

    public static Tensor broadcast(Tensor a, Tensor b, Op op) {
        int[] os = broadcastShape(a.shape, b.shape);
        Tensor out = zeros(os);
        int n = os.length;
        int[] idx = new int[n];
        for (int flat = 0; flat < out.size(); flat++) {
            // decode flat -> multi-index of the output
            int rem = flat;
            for (int d = 0; d < n; d++) { idx[d] = rem / out.strides[d]; rem %= out.strides[d]; }
            out.data[flat] = op.apply(a.data[bOffset(a, idx, n)], b.data[bOffset(b, idx, n)]);
        }
        return out;
    }

    /** Offset into t for an output index; a size-1 dimension always uses index 0 ("stretched"). */
    private static int bOffset(Tensor t, int[] outIdx, int n) {
        int o = 0, shift = n - t.shape.length;
        for (int d = 0; d < t.shape.length; d++) {
            int i = t.shape[d] == 1 ? 0 : outIdx[d + shift];
            o += i * t.strides[d];
        }
        return o;
    }

    public Tensor add(Tensor o) { return broadcast(this, o, (x, y) -> x + y); }
    public Tensor sub(Tensor o) { return broadcast(this, o, (x, y) -> x - y); }
    public Tensor mul(Tensor o) { return broadcast(this, o, (x, y) -> x * y); }

    public Tensor scale(double k) {
        Tensor t = copy();
        for (int i = 0; i < t.data.length; i++) t.data[i] *= k;
        return t;
    }

    public interface Fn { double apply(double x); }

    public Tensor map(Fn f) {
        Tensor t = copy();
        for (int i = 0; i < t.data.length; i++) t.data[i] = f.apply(t.data[i]);
        return t;
    }

    public double sum() { double s = 0; for (double v : data) s += v; return s; }

    /** Sum over rows of a 2-D tensor -> shape (1, cols). Used for bias gradients. */
    public Tensor sumRows() {
        check2d();
        Tensor t = zeros(1, shape[1]);
        for (int i = 0; i < shape[0]; i++)
            for (int j = 0; j < shape[1]; j++) t.data[j] += data[i * shape[1] + j];
        return t;
    }

    // ------------------------------------------------------------ inner products
    /** Inner (dot) product of two tensors with the same number of elements: sum_i a_i * b_i */
    public static double dot(Tensor a, Tensor b) {
        if (a.size() != b.size()) throw new IllegalArgumentException("dot: size mismatch");
        double s = 0;
        for (int i = 0; i < a.size(); i++) s += a.data[i] * b.data[i];
        return s;
    }

    public double norm() { return Math.sqrt(dot(this, this)); }

    /** Matrix product (m,k) x (k,n) -> (m,n). Each output = inner product of a row and a column. */
    public Tensor matmul(Tensor o) {
        check2d(); o.check2d();
        int m = shape[0], k = shape[1], n = o.shape[1];
        if (o.shape[0] != k)
            throw new IllegalArgumentException("matmul: " + Arrays.toString(shape) + " x " + Arrays.toString(o.shape));
        Tensor out = zeros(m, n);
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++) {
                double a = data[i * k + p];
                for (int j = 0; j < n; j++) out.data[i * n + j] += a * o.data[p * n + j];
            }
        return out;
    }

    void check2d() { if (shape.length != 2) throw new IllegalStateException("expected 2-D tensor, got " + Arrays.toString(shape)); }

    @Override
    public String toString() {
        if (shape.length == 2 && size() <= 64) {
            StringBuilder sb = new StringBuilder("Tensor" + Arrays.toString(shape) + "\n");
            for (int i = 0; i < shape[0]; i++) {
                sb.append("  [");
                for (int j = 0; j < shape[1]; j++) sb.append(String.format("%8.3f", get(i, j)));
                sb.append(" ]\n");
            }
            return sb.toString();
        }
        return "Tensor" + Arrays.toString(shape) + (size() <= 16 ? " " + Arrays.toString(data) : "");
    }
}