package dl;

import java.util.Random;

/**
 * Hopfield network — an energy-based associative memory.
 *
 * States s_i ∈ {−1, +1}.  Hebbian storage of P patterns:  W = (1/N) Σ_p  ξ_p ξ_p^T ,  W_ii = 0
 * Energy:   E(s) = −½ Σ_i Σ_j W_ij s_i s_j
 * Update (one neuron at a time):  s_i ← sign( Σ_j W_ij s_j )
 * Each update can only keep E the same or LOWER it, so the network rolls "downhill"
 * into the nearest stored pattern (an energy minimum).  Corrupted input -> clean memory.
 *
 * Link to anomaly detection: normal patterns = low energy; an input far from all stored
 * memories has high energy — the same "distance from normal" idea PatchCore uses.
 */
public class Hopfield {
    final int n;
    final double[][] W;

    public Hopfield(int n) { this.n = n; this.W = new double[n][n]; }

    public void store(int[][] patterns) {
        for (int[] p : patterns)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (i != j) W[i][j] += p[i] * p[j] / (double) n;
    }

    public double energy(int[] s) {
        double e = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) e += W[i][j] * s[i] * s[j];
        return -0.5 * e;
    }

    /** Asynchronous recall. Returns energy after every single-neuron update. */
    public double[] recall(int[] s, int sweeps, long seed) {
        Random r = new Random(seed);
        double[] trace = new double[sweeps * n + 1];
        trace[0] = energy(s);
        int k = 1;
        for (int sw = 0; sw < sweeps; sw++)
            for (int t = 0; t < n; t++) {
                int i = r.nextInt(n);
                double h = 0;
                for (int j = 0; j < n; j++) h += W[i][j] * s[j];
                s[i] = h >= 0 ? 1 : -1;
                trace[k++] = energy(s);
            }
        return trace;
    }

    public static int[] fromAscii(String... rows) {
        int[] p = new int[rows.length * rows[0].length()];
        for (int i = 0; i < rows.length; i++)
            for (int j = 0; j < rows[i].length(); j++) p[i * rows[0].length() + j] = rows[i].charAt(j) == '#' ? 1 : -1;
        return p;
    }

    public static int[] corrupt(int[] p, double frac, long seed) {
        Random r = new Random(seed);
        int[] s = p.clone();
        for (int i = 0; i < s.length; i++) if (r.nextDouble() < frac) s[i] = -s[i];
        return s;
    }

    public static int hamming(int[] a, int[] b) {
        int d = 0;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) d++;
        return d;
    }

    public static Tensor toImage(int[] s, int side) {
        double[] d = new double[s.length];
        for (int i = 0; i < s.length; i++) d[i] = s[i];
        return new Tensor(d, side, side);
    }
}