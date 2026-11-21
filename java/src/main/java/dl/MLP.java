package dl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Multilayer feed-forward network for binary classification.
 *
 * Shapes (N = batch size):
 *   X      (N, d0)
 *   Z_l = A_{l-1} W_l + b_l      W_l (d_{l-1}, d_l),  b_l (1, d_l)   <- b is BROADCAST over the N rows
 *   A_l = act(Z_l)               hidden: tanh or sigmoid;  output: sigmoid
 *   Loss = mean BCE + (lambda/2) * sum ||W_l||^2
 *
 * Backpropagation (chain rule, applied from the last layer backwards):
 *   dZ_L = (A_L - Y) / N                     (sigmoid + BCE simplify to this)
 *   dW_l = A_{l-1}^T dZ_l + lambda W_l
 *   db_l = sum over rows of dZ_l
 *   dA_{l-1} = dZ_l W_l^T
 *   dZ_{l-1} = dA_{l-1} * act'(Z_{l-1})       (* = element-wise)
 */
public class MLP {
    public enum Act { SIGMOID, TANH, RELU }
    public enum Init { ZERO, SMALL_NORMAL, XAVIER, HE }

    public final int[] sizes;
    public final Act hiddenAct;
    public final List<Tensor> W = new ArrayList<>(), b = new ArrayList<>();
    public final List<Tensor> dW = new ArrayList<>(), db = new ArrayList<>();
    public double l2 = 0.0;

    // caches from the last forward pass, needed by backward
    private final List<Tensor> Zs = new ArrayList<>(), As = new ArrayList<>();

    public MLP(int[] sizes, Act hiddenAct, Init init, long seed) {
        this.sizes = sizes; this.hiddenAct = hiddenAct;
        Random rng = new Random(seed);
        for (int l = 1; l < sizes.length; l++) {
            int fanIn = sizes[l - 1], fanOut = sizes[l];
            double std = switch (init) {
                case ZERO -> 0.0;
                case SMALL_NORMAL -> 0.01;
                case XAVIER -> Math.sqrt(2.0 / (fanIn + fanOut));   // keeps variance stable for tanh/sigmoid
                case HE -> Math.sqrt(2.0 / fanIn);                   // keeps variance stable for ReLU
            };
            W.add(Tensor.randn(rng, std, fanIn, fanOut));
            b.add(Tensor.zeros(1, fanOut));
            dW.add(Tensor.zeros(fanIn, fanOut));
            db.add(Tensor.zeros(1, fanOut));
        }
    }

    double act(double z) {
        return switch (hiddenAct) {
            case SIGMOID -> Neurons.sigmoid(z);
            case TANH -> Math.tanh(z);
            case RELU -> Math.max(0, z);
        };
    }

    double actGrad(double z) {
        return switch (hiddenAct) {
            case SIGMOID -> { double s = Neurons.sigmoid(z); yield s * (1 - s); }
            case TANH -> { double t = Math.tanh(z); yield 1 - t * t; }
            case RELU -> z > 0 ? 1 : 0;
        };
    }

    /** Forward propagation. Returns predicted probabilities, shape (N, 1). */
    public Tensor forward(Tensor X) {
        Zs.clear(); As.clear();
        As.add(X);
        Tensor A = X;
        for (int l = 0; l < W.size(); l++) {
            Tensor Z = A.matmul(W.get(l)).add(b.get(l));          // broadcasting adds the bias row to every sample
            boolean last = l == W.size() - 1;
            A = last ? Z.map(Neurons::sigmoid) : Z.map(this::act);
            Zs.add(Z); As.add(A);
        }
        return A;
    }

    /** Mean binary cross-entropy + L2 penalty. */
    public double loss(Tensor P, Tensor Y) {
        double s = 0;
        for (int i = 0; i < P.size(); i++) {
            double p = Math.min(Math.max(P.data[i], 1e-12), 1 - 1e-12);
            s += -(Y.data[i] * Math.log(p) + (1 - Y.data[i]) * Math.log(1 - p));
        }
        double reg = 0;
        for (Tensor w : W) reg += Tensor.dot(w, w);
        return s / P.size() + 0.5 * l2 * reg;
    }

    /** Backpropagation. Must be called right after forward(X). Fills dW and db. */
    public void backward(Tensor Y) {
        int N = Y.shape[0];
        int L = W.size();
        Tensor dZ = As.get(L).sub(Y).scale(1.0 / N);
        for (int l = L - 1; l >= 0; l--) {
            Tensor Aprev = As.get(l);
            Tensor gW = Aprev.transpose().matmul(dZ).add(W.get(l).scale(l2));
            Tensor gb = dZ.sumRows();
            System.arraycopy(gW.data, 0, dW.get(l).data, 0, gW.size());
            System.arraycopy(gb.data, 0, db.get(l).data, 0, gb.size());
            if (l > 0) {
                Tensor dA = dZ.matmul(W.get(l).transpose());
                dZ = dA.mul(Zs.get(l - 1).map(this::actGrad));
            }
        }
    }

    public List<Tensor> params() { List<Tensor> p = new ArrayList<>(W); p.addAll(b); return p; }
    public List<Tensor> grads()  { List<Tensor> g = new ArrayList<>(dW); g.addAll(db); return g; }

    public double accuracy(Tensor X, Tensor Y) {
        Tensor P = forward(X);
        int ok = 0;
        for (int i = 0; i < P.size(); i++) if ((P.data[i] >= 0.5 ? 1 : 0) == (int) Y.data[i]) ok++;
        return ok / (double) P.size();
    }

    public double predict1(double x, double y) {
        return forward(new Tensor(new double[]{x, y}, 1, 2)).data[0];
    }
}