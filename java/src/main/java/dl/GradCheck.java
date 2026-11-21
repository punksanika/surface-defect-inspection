package dl;

import java.util.function.Supplier;

/**
 * Numerical gradient checking.
 *
 * Central difference:   dL/dθ_i ≈ [ L(θ_i + h) - L(θ_i - h) ] / (2h)
 * Error is O(h^2), so with h = 1e-5 the estimate is very accurate.
 *
 * We compare it with the ANALYTIC gradient from backpropagation using the
 * relative error  |g_num - g_ana| / max(|g_num| + |g_ana|, tiny).
 * Rule of thumb: < 1e-7 excellent, < 1e-5 OK, > 1e-3 there is a bug.
 */
public class GradCheck {

    /** Numerically differentiate loss() with respect to every element of param (param is modified in place and restored). */
    public static Tensor numericalGrad(Tensor param, Supplier<Double> loss, double h) {
        Tensor g = Tensor.zeros(param.shape);
        for (int i = 0; i < param.size(); i++) {
            double old = param.data[i];
            param.data[i] = old + h; double lp = loss.get();
            param.data[i] = old - h; double lm = loss.get();
            param.data[i] = old;
            g.data[i] = (lp - lm) / (2 * h);
        }
        return g;
    }

    public static double maxRelativeError(Tensor numeric, Tensor analytic) {
        double worst = 0;
        for (int i = 0; i < numeric.size(); i++) {
            double a = numeric.data[i], b = analytic.data[i];
            double rel = Math.abs(a - b) / Math.max(Math.abs(a) + Math.abs(b), 1e-12);
            if (Math.abs(a) + Math.abs(b) < 1e-12) rel = 0;   // both essentially zero
            worst = Math.max(worst, rel);
        }
        return worst;
    }
}