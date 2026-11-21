package dl;

import java.util.ArrayList;
import java.util.List;

/**
 * Gradient-descent optimizers. θ = parameters, g = gradient, η = learning rate, μ = momentum.
 *
 *  GD (used for both Batch GD and SGD — they differ only in how many samples produce g):
 *        θ ← θ − η g
 *  Momentum (heavy ball): velocity accumulates past gradients, smoothing zig-zags
 *        v ← μ v + g ;   θ ← θ − η v
 *  Nesterov: "look ahead" — use the gradient direction after the momentum step
 *        v ← μ v + g ;   θ ← θ − η (g + μ v)          (the form used by PyTorch)
 */
public class Optimizer {
    public enum Type { GD, MOMENTUM, NESTEROV }

    final Type type; final double lr, mu;
    private final List<double[]> velocity = new ArrayList<>();

    public Optimizer(Type type, double lr, double mu) { this.type = type; this.lr = lr; this.mu = mu; }

    public void step(List<Tensor> params, List<Tensor> grads) {
        if (velocity.isEmpty()) for (Tensor p : params) velocity.add(new double[p.size()]);
        for (int k = 0; k < params.size(); k++) {
            double[] p = params.get(k).data, g = grads.get(k).data, v = velocity.get(k);
            for (int i = 0; i < p.length; i++) {
                switch (type) {
                    case GD -> p[i] -= lr * g[i];
                    case MOMENTUM -> { v[i] = mu * v[i] + g[i]; p[i] -= lr * v[i]; }
                    case NESTEROV -> { v[i] = mu * v[i] + g[i]; p[i] -= lr * (g[i] + mu * v[i]); }
                }
            }
        }
    }

    @Override public String toString() { return type + "(lr=" + lr + (type == Type.GD ? "" : ", mu=" + mu) + ")"; }
}