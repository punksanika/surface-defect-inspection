package dl;

import java.util.Random;

/**
 * Classical learning rules. y = w . x is a single linear neuron unless stated otherwise.
 *
 *  Hebbian           Δw = η y x                 "neurons that fire together wire together"; ||w|| grows without bound
 *  Oja (modified)    Δw = η y (x − y w)         Hebb + decay term; ||w|| -> 1 and w -> FIRST PRINCIPAL COMPONENT
 *  Competitive       winner k = argmin_j ||x − w_j||,  Δw_k = η (x − w_k)   (only the winner learns)
 *  Error-correction  Δw = η (t − y) x           (delta / perceptron rule: learn from the ERROR, the root of backprop)
 */
public class LearningRules {

    /** 2-D Gaussian data stretched along angle `deg` (so it has a clear principal direction). Zero-mean. */
    public static double[][] correlatedData(int n, double deg, long seed) {
        Random r = new Random(seed);
        double a = Math.toRadians(deg), c = Math.cos(a), s = Math.sin(a);
        double[][] x = new double[n][2];
        for (int i = 0; i < n; i++) {
            double u = 3.0 * r.nextGaussian(), v = 0.5 * r.nextGaussian();
            x[i][0] = c * u - s * v; x[i][1] = s * u + c * v;
        }
        return x;
    }

    /** Returns history of ||w|| for each epoch; final w written into w. */
    public static double[] hebbian(double[][] X, double[] w, double eta, int epochs) {
        double[] norms = new double[epochs];
        for (int e = 0; e < epochs; e++) {
            for (double[] x : X) {
                double y = w[0] * x[0] + w[1] * x[1];
                w[0] += eta * y * x[0]; w[1] += eta * y * x[1];
            }
            norms[e] = Math.hypot(w[0], w[1]);
        }
        return norms;
    }

    public static double[] oja(double[][] X, double[] w, double eta, int epochs) {
        double[] norms = new double[epochs];
        for (int e = 0; e < epochs; e++) {
            for (double[] x : X) {
                double y = w[0] * x[0] + w[1] * x[1];
                w[0] += eta * y * (x[0] - y * w[0]); w[1] += eta * y * (x[1] - y * w[1]);
            }
            norms[e] = Math.hypot(w[0], w[1]);
        }
        return norms;
    }

    /** First principal direction from our own SVD (rows = samples). */
    public static double[] pcaDirection(double[][] X) {
        Tensor A = Tensor.of2d(X);
        Linalg.SVD s = Linalg.svd(A);
        return new double[]{s.V.get(0, 0), s.V.get(1, 0)};
    }

    public static double absCosine(double[] a, double[] b) {
        return Math.abs(a[0] * b[0] + a[1] * b[1]) / (Math.hypot(a[0], a[1]) * Math.hypot(b[0], b[1]));
    }

    // ------------------------------------------------------------ competitive learning
    public static double[][] blobs(int nPer, double[][] centers, double std, long seed, int[] labelsOut) {
        Random r = new Random(seed);
        double[][] x = new double[nPer * centers.length][2];
        for (int c = 0; c < centers.length; c++)
            for (int i = 0; i < nPer; i++) {
                int k = c * nPer + i;
                x[k][0] = centers[c][0] + std * r.nextGaussian();
                x[k][1] = centers[c][1] + std * r.nextGaussian();
                if (labelsOut != null) labelsOut[k] = c;
            }
        return x;
    }

    /** Winner-take-all. Prototypes initialised from random samples (avoids "dead" units). */
    public static double[][] competitive(double[][] X, int k, double eta, int epochs, long seed) {
        Random r = new Random(seed);
        double[][] w = new double[k][];
        for (int j = 0; j < k; j++) w[j] = X[r.nextInt(X.length)].clone();
        for (int e = 0; e < epochs; e++)
            for (int t = 0; t < X.length; t++) {
                double[] x = X[r.nextInt(X.length)];
                int win = winner(w, x);
                w[win][0] += eta * (x[0] - w[win][0]);
                w[win][1] += eta * (x[1] - w[win][1]);
            }
        return w;
    }

    public static int winner(double[][] w, double[] x) {
        int best = 0; double bd = Double.MAX_VALUE;
        for (int j = 0; j < w.length; j++) {
            double d = Math.pow(x[0] - w[j][0], 2) + Math.pow(x[1] - w[j][1], 2);
            if (d < bd) { bd = d; best = j; }
        }
        return best;
    }

    // ------------------------------------------------------------ error-correction (perceptron)
    /** Perceptron with threshold output; returns number of mistakes per epoch. w has 3 entries: w0 (bias), w1, w2. */
    public static int[] perceptron(double[][] X, int[] t, double[] w, double eta, int epochs) {
        int[] mistakes = new int[epochs];
        for (int e = 0; e < epochs; e++)
            for (int i = 0; i < X.length; i++) {
                int y = w[0] + w[1] * X[i][0] + w[2] * X[i][1] >= 0 ? 1 : 0;
                int err = t[i] - y;                 // 0 when correct -> no update
                if (err != 0) mistakes[e]++;
                w[0] += eta * err; w[1] += eta * err * X[i][0]; w[2] += eta * err * X[i][1];
            }
        return mistakes;
    }
}