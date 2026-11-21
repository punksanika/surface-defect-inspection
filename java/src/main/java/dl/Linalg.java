package dl;

/**
 * Singular Value Decomposition by one-sided Jacobi rotations.
 *
 *   A (m x n) = U * diag(S) * V^T
 *
 * Idea: keep rotating PAIRS of columns of A until every pair of columns is orthogonal.
 * Each rotation is also applied to V (starting from identity). At the end:
 *   column norms of the rotated A  = singular values S
 *   rotated A / S                  = U
 *   accumulated rotations          = V
 */
public class Linalg {

    public static class SVD {
        public final Tensor U, S, V;   // U (m x n), S (n), V (n x n)
        SVD(Tensor U, Tensor S, Tensor V) { this.U = U; this.S = S; this.V = V; }

        /** Rebuild A using only the top-k singular values (low-rank approximation). */
        public Tensor reconstruct(int k) {
            int m = U.shape[0], n = V.shape[0];
            Tensor A = Tensor.zeros(m, n);
            for (int r = 0; r < k; r++)
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        A.data[i * n + j] += S.data[r] * U.get(i, r) * V.get(j, r);
            return A;
        }
    }

    public static SVD svd(Tensor A) {
        A.check2d();
        int m = A.shape[0], n = A.shape[1];
        if (m < n) throw new IllegalArgumentException("this simple SVD expects rows >= cols; transpose first");
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) a[i][j] = A.get(i, j);
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;

        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            for (int p = 0; p < n - 1; p++)
                for (int q = p + 1; q < n; q++) {
                    double alpha = 0, beta = 0, gamma = 0;   // ||a_p||^2, ||a_q||^2, a_p . a_q
                    for (int i = 0; i < m; i++) {
                        alpha += a[i][p] * a[i][p];
                        beta += a[i][q] * a[i][q];
                        gamma += a[i][p] * a[i][q];
                    }
                    if (Math.abs(gamma) < 1e-15) continue;
                    off = Math.max(off, Math.abs(gamma) / Math.sqrt(alpha * beta));
                    // rotation angle that makes columns p and q orthogonal
                    double zeta = (beta - alpha) / (2 * gamma);
                    double t = Math.signum(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
                    if (zeta == 0) t = 1;
                    double c = 1 / Math.sqrt(1 + t * t), s = c * t;
                    for (int i = 0; i < m; i++) {
                        double ap = a[i][p], aq = a[i][q];
                        a[i][p] = c * ap - s * aq; a[i][q] = s * ap + c * aq;
                    }
                    for (int i = 0; i < n; i++) {
                        double vp = v[i][p], vq = v[i][q];
                        v[i][p] = c * vp - s * vq; v[i][q] = s * vp + c * vq;
                    }
                }
            if (off < 1e-12) break;
        }
        // singular values = column norms; sort descending
        double[] sv = new double[n];
        Integer[] order = new Integer[n];
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int i = 0; i < m; i++) s += a[i][j] * a[i][j];
            sv[j] = Math.sqrt(s); order[j] = j;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(sv[y], sv[x]));
        Tensor U = Tensor.zeros(m, n), S = Tensor.zeros(n), V = Tensor.zeros(n, n);
        for (int k = 0; k < n; k++) {
            int j = order[k];
            S.data[k] = sv[j];
            for (int i = 0; i < m; i++) U.set(sv[j] > 1e-15 ? a[i][j] / sv[j] : 0, i, k);
            for (int i = 0; i < n; i++) V.set(v[i][j], i, k);
        }
        return new SVD(U, S, V);
    }

    /** Frobenius norm of (A - B): how different two matrices are. */
    public static double frobDiff(Tensor A, Tensor B) { return A.sub(B).norm(); }
}