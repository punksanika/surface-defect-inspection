"""Run every fundamentals experiment, save all figures, and print a results summary.

    python -m fundamentals.run_all

Figures and per-run logs: experiments/fundamentals/   Summary: experiments/fundamentals/summary.json
"""
import json
import time

from fundamentals import learning_rules, mlp, neurons, tensor_sandbox, training_experiments
from fundamentals.common import OUT


def main():
    t0 = time.time()
    results = {
        "tensor_sandbox": tensor_sandbox.run(),
        "neurons": neurons.run(),
        "mlp": mlp.run(),
        "training_experiments": training_experiments.run(),
        "learning_rules": learning_rules.run(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))

    sb, ne, ml = results["tensor_sandbox"], results["neurons"], results["mlp"]
    te, lr = results["training_experiments"], results["learning_rules"]
    print("\n" + "=" * 30 + " RESULTS SUMMARY " + "=" * 30)
    print("Req 3  Tensor/broadcasting : dot(u,v) = %g; incompatible shapes -> error" % sb["dot_u_v"])
    print("       SVD (Jacobi)        : 64x64 max |S - S_numpy| = %.2e; rel. error rank 1/5/20 = %.3f / %.3f / %.3f"
          % (sb["svd_64_max_diff_vs_numpy"], sb["svd_64_rel_error_rank1"], sb["svd_64_rel_error_rank5"],
             sb["svd_64_rel_error_rank20"]))
    print("       Gradient check      : max rel. error %.2e at h=1e-5 (best h = %.0e)"
          % (sb["gradcheck_rel_error_h1e-5"], sb["gradcheck_best_h"]))
    print("Req 4  McCulloch-Pitts     : XOR solutions on the grid = %d of %d"
          % (ne["xor_grid_solutions_found"], ne["xor_grid_settings_tried"]))
    print("       Sigmoid neuron (OR) : loss %.4f -> %.4f" % (ne["sigmoid_OR_loss_first"], ne["sigmoid_OR_loss_last"]))
    print("       MLP backprop check  : max rel. error %.2e (tanh), %.2e (sigmoid + L2)"
          % (ml["backprop_gradcheck_max_rel_error"], ml["backprop_gradcheck_max_rel_error_sigmoid_l2"]))
    print("       MLP on XOR          : accuracy %.2f, outputs %s" % (ml["xor_accuracy"], ml["xor_outputs"]))
    print("Req 5  Optimizers (val acc): " + ", ".join(f"{r['optimizer']} {r['val_acc']:.3f}" for r in te["optimizers"]))
    print("       Initialisation      : " + ", ".join(f"{r['init']} {r['val_acc']:.3f}" for r in te["initialisation"]))
    print("       Generalisation      : " + ", ".join(f"{r['setup']} train {r['train_acc']:.3f}/val {r['val_acc']:.3f}"
                                                   for r in te["generalisation"]))
    print("       Standardisation     : " + ", ".join(f"{r['input']} {r['val_acc']:.3f}" for r in te["normalisation"]))
    print("Req 6  Hebbian             : ||w|| = %.3e (explodes)" % lr["hebbian_final_norm"])
    print("       Oja vs own-SVD PC1  : ||w|| = %.4f, |cos| = %.4f" % (lr["oja_final_norm"], lr["oja_abs_cos_pc1"]))
    print("       Competitive         : worst prototype-to-centre distance %.3f" % lr["competitive_max_centre_error"])
    print("       Perceptron          : mistakes per epoch %s" % lr["perceptron_mistakes_per_epoch"])
    print("       Hopfield            : " + "; ".join(
        f"{h['pattern']} {h['wrong_pixels_before']}->{h['wrong_pixels_after']} wrong px" for h in lr["hopfield"])
          + "; energy never increased: %s" % all(h["energy_never_increased"] for h in lr["hopfield"]))
    print("Figures: %s   (%.1f s)" % (OUT.relative_to(OUT.parent.parent), time.time() - t0))


if __name__ == "__main__":
    main()
