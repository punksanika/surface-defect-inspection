package dl;

import java.util.Random;

/** Small synthetic datasets (fixed seeds -> reproducible). */
public class Data {
    public final Tensor X, Y;   // X (N,2), Y (N,1)
    public Data(Tensor X, Tensor Y) { this.X = X; this.Y = Y; }

    public static Data xor() {
        return new Data(Tensor.of2d(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}),
                new Tensor(new double[]{0, 1, 1, 0}, 4, 1));
    }

    /** Two interleaving half-circles: not linearly separable, needs a hidden layer. */
    public static Data moons(int n, double noise, long seed) {
        Random r = new Random(seed);
        double[] x = new double[2 * n], y = new double[n];
        for (int i = 0; i < n; i++) {
            double t = Math.PI * r.nextDouble();
            boolean upper = i % 2 == 0;
            x[2 * i] = upper ? Math.cos(t) : 1 - Math.cos(t);
            x[2 * i + 1] = upper ? Math.sin(t) : 0.5 - Math.sin(t);
            x[2 * i] += noise * r.nextGaussian();
            x[2 * i + 1] += noise * r.nextGaussian();
            y[i] = upper ? 0 : 1;
        }
        return new Data(new Tensor(x, n, 2), new Tensor(y, n, 1));
    }

    /** Rows [from, to) as a new dataset (mini-batch). */
    public Data slice(int[] order, int from, int to) {
        int d = X.shape[1], m = to - from;
        double[] xs = new double[m * d], ys = new double[m];
        for (int i = 0; i < m; i++) {
            int r = order[from + i];
            System.arraycopy(X.data, r * d, xs, i * d, d);
            ys[i] = Y.data[r];
        }
        return new Data(new Tensor(xs, m, d), new Tensor(ys, m, 1));
    }

    public int n() { return X.shape[0]; }

    public double[][] points() {
        double[][] p = new double[n()][2];
        for (int i = 0; i < n(); i++) { p[i][0] = X.get(i, 0); p[i][1] = X.get(i, 1); }
        return p;
    }

    public int[] labels() {
        int[] c = new int[n()];
        for (int i = 0; i < n(); i++) c[i] = (int) Y.data[i];
        return c;
    }

    /** Column means and stds computed on THIS dataset (call on train only -> no leakage). */
    public double[][] stats() {
        int d = X.shape[1]; double[] mean = new double[d], std = new double[d];
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n(); i++) mean[j] += X.get(i, j);
            mean[j] /= n();
            for (int i = 0; i < n(); i++) std[j] += Math.pow(X.get(i, j) - mean[j], 2);
            std[j] = Math.sqrt(std[j] / n()) + 1e-12;
        }
        return new double[][]{mean, std};
    }

    /** Standardize: (x - mean) / std using broadcasting of a (1,d) row over (N,d). */
    public Data standardize(double[][] s) {
        Tensor mean = new Tensor(s[0].clone(), 1, s[0].length), std = new Tensor(s[1].clone(), 1, s[1].length);
        return new Data(Tensor.broadcast(X.sub(mean), std, (a, b) -> a / b), Y);
    }

    /** Augmentation: k jittered copies of every sample (Gaussian noise), labels unchanged. */
    public Data augment(int k, double sigma, long seed) {
        Random r = new Random(seed);
        int n = n(), d = X.shape[1];
        double[] xs = new double[n * (k + 1) * d], ys = new double[n * (k + 1)];
        for (int c = 0; c <= k; c++)
            for (int i = 0; i < n; i++) {
                int row = c * n + i;
                for (int j = 0; j < d; j++) xs[row * d + j] = X.get(i, j) + (c == 0 ? 0 : sigma * r.nextGaussian());
                ys[row] = Y.data[i];
            }
        return new Data(new Tensor(xs, n * (k + 1), d), new Tensor(ys, n * (k + 1), 1));
    }
}