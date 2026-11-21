package dl;

/** Single neurons: McCulloch-Pitts (1943) and the sigmoid neuron. */
public class Neurons {

    /**
     * McCulloch-Pitts neuron: binary inputs, fixed weights, hard threshold.
     *    y = 1  if  sum_i w_i x_i >= theta  else 0
     * Nothing is learned; the designer chooses w and theta.
     */
    public static class McCullochPitts {
        final double[] w; final double theta;
        public McCullochPitts(double[] w, double theta) { this.w = w; this.theta = theta; }
        public int fire(int... x) {
            double s = 0;
            for (int i = 0; i < x.length; i++) s += w[i] * x[i];
            return s >= theta ? 1 : 0;
        }
        public static McCullochPitts AND() { return new McCullochPitts(new double[]{1, 1}, 2); }
        public static McCullochPitts OR()  { return new McCullochPitts(new double[]{1, 1}, 1); }
        public static McCullochPitts NOT() { return new McCullochPitts(new double[]{-1}, 0); }
    }

    /**
     * Try every weight/threshold on a grid: no single threshold neuron computes XOR,
     * because XOR is not linearly separable (no straight line splits its 1s from its 0s).
     * Returns the number of grid settings that solve XOR (expected: 0).
     */
    public static int searchXorSolutions() {
        int[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] Y = {0, 1, 1, 0};
        int found = 0;
        for (double w1 = -3; w1 <= 3; w1 += 0.5)
            for (double w2 = -3; w2 <= 3; w2 += 0.5)
                for (double th = -3; th <= 3; th += 0.5) {
                    McCullochPitts n = new McCullochPitts(new double[]{w1, w2}, th);
                    boolean ok = true;
                    for (int i = 0; i < 4; i++) if (n.fire(X[i]) != Y[i]) { ok = false; break; }
                    if (ok) found++;
                }
        return found;
    }

    public static double sigmoid(double z) { return 1.0 / (1.0 + Math.exp(-z)); }

    /**
     * Sigmoid neuron: y_hat = sigmoid(w.x + b), a smooth, differentiable version of the threshold.
     * Trained with binary cross-entropy  L = -[y log y_hat + (1-y) log(1-y_hat)].
     * Gradient:  dL/dz = y_hat - y,  dL/dw = (y_hat - y) x,  dL/db = y_hat - y
     */
    public static class SigmoidNeuron {
        public double[] w; public double b;
        public SigmoidNeuron(int nIn) { w = new double[nIn]; }

        public double forward(double[] x) {
            double z = b;
            for (int i = 0; i < x.length; i++) z += w[i] * x[i];
            return sigmoid(z);
        }

        /** One epoch of full-batch gradient descent. Returns mean BCE loss before the update. */
        public double trainEpoch(double[][] X, double[] Y, double lr) {
            double[] gw = new double[w.length]; double gb = 0, loss = 0;
            for (int n = 0; n < X.length; n++) {
                double p = forward(X[n]);
                loss += -(Y[n] * Math.log(p + 1e-12) + (1 - Y[n]) * Math.log(1 - p + 1e-12));
                double dz = p - Y[n];
                for (int i = 0; i < w.length; i++) gw[i] += dz * X[n][i];
                gb += dz;
            }
            for (int i = 0; i < w.length; i++) w[i] -= lr * gw[i] / X.length;
            b -= lr * gb / X.length;
            return loss / X.length;
        }
    }
}