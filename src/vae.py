"""
vae.py: convolutional Variational Autoencoder on GOOD nuts + latent-space visualizations + defect synthesizer.

Run:
    python src/vae.py

VAE in one paragraph:
  Encoder q(z|x): image (3,128,128) -> mean mu and log-variance logvar of a 64-dim Gaussian.
  Reparameterization: z = mu + sigma * eps, eps ~ N(0, I)   (randomness moved outside -> gradients flow)
  Decoder p(x|z): z -> reconstructed image.
  Loss (negative ELBO) = reconstruction error + beta * KL( q(z|x) || N(0,I) )
     KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))   pulls codes towards a smooth, filled latent space

Outputs (experiments/vae/):
  loss_curve.png         training loss
  reconstructions.png    test good / defective nuts and their reconstructions (defects are "repaired")
  interpolation.png      walk in latent space between two good nuts      <- latent-space visualization
  samples.png            new nuts generated from random z ~ N(0,I)
  latent_pca.png         2-D PCA of mu for test good vs defective nuts   <- latent-space visualization
  synthetic_defects.png  defect synthesizer: real defect regions (from sup_train masks) composited onto
                         VAE-generated normal nuts -> new labelled defect images + masks
  metrics.json           VAE reconstruction-error anomaly AUROC (baseline vs PatchCore)
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision.transforms import functional as TF

S_IMG, ZDIM, BETA = 128, 64, 1.0
OUT = Path("experiments/vae")


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(path, size=S_IMG):
    return TF.to_tensor(Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR))


def load_mask(path, size=S_IMG):
    return (TF.to_tensor(Image.open(path).convert("L").resize((size, size), Image.NEAREST)) > 0.5).float()


class VAE(nn.Module):
    def __init__(self, z=ZDIM):
        super().__init__()
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc = nn.Sequential(down(3, 32), down(32, 64), down(64, 128), down(128, 256))   # 128 -> 8
        self.fc_mu = nn.Linear(256 * 8 * 8, z)
        self.fc_lv = nn.Linear(256 * 8 * 8, z)
        self.fc_dec = nn.Linear(z, 256 * 8 * 8)
        self.dec = nn.Sequential(up(256, 128), up(128, 64), up(64, 32),
                                 nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())          # 8 -> 128

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        return self.dec(self.fc_dec(z).view(-1, 256, 8, 8))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)        # reparameterization trick
        return self.decode(z), mu, lv


def vae_loss(xr, x, mu, lv):
    rec = F.mse_loss(xr, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / x.shape[0]
    return rec + BETA * kl, rec, kl


def grid(rows, path, titles=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, C = len(rows), len(rows[0])
    fig, axes = plt.subplots(R, C, figsize=(1.6 * C, 1.7 * R))
    axes = np.array(axes).reshape(R, C)
    for i in range(R):
        for j in range(C):
            img = rows[i][j]
            axes[i, j].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy() if img.shape[0] == 3 else img[0].numpy(),
                              cmap=None if img.shape[0] == 3 else "gray")
            axes[i, j].axis("off")
        if titles:
            axes[i, 0].set_title(titles[i], fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = get_device(); print("Device:", dev)
    S = json.loads(Path("splits.json").read_text())["splits"]
    X = torch.stack([load(r["path"]) for r in S["train_good"]])                 # (190,3,128,128)
    print("Training images:", tuple(X.shape))

    model = VAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs, bs, hist = 150, 16, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = rec_t = kl_t = 0.0
        for k in range(0, len(X), bs):
            xb = X[perm[k:k + bs]].to(dev)
            xr, mu, lv = model(xb)
            loss, rec, kl = vae_loss(xr, xb, mu, lv)
            opt.zero_grad(); loss.backward(); opt.step()
            n = len(xb); tot += loss.item() * n; rec_t += rec.item() * n; kl_t += kl.item() * n
        hist.append([tot / len(X), rec_t / len(X), kl_t / len(X)])
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d}  loss {hist[-1][0]:8.1f}  (reconstruction {hist[-1][1]:8.1f}  KL {hist[-1][2]:6.1f})")
    print(f"Training time {time.time() - t0:.0f}s")
    torch.save(model.state_dict(), OUT / "vae.pt")
    model.eval()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = np.array(hist)
    plt.figure(figsize=(6, 3.5)); plt.plot(h[:, 0], label="total"); plt.plot(h[:, 1], label="reconstruction")
    plt.plot(h[:, 2], label="KL"); plt.yscale("log"); plt.xlabel("epoch"); plt.legend(); plt.title("VAE training loss")
    plt.tight_layout(); plt.savefig(OUT / "loss_curve.png", dpi=110); plt.close()

    with torch.no_grad():
        # --- reconstruction error as anomaly score on TEST (baseline to compare with PatchCore)
        test = S["test"]
        xt = torch.stack([load(r["path"]) for r in test]).to(dev)
        mu_t, _ = model.encode(xt)
        rec_t = model.decode(mu_t)
        err = ((rec_t - xt) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        y = np.array([0 if r["label"] == "good" else 1 for r in test])
        auroc = float(roc_auc_score(y, err))
        print(f"VAE reconstruction-error anomaly AUROC on test: {auroc:.3f}  (PatchCore ResNet-18: 0.974)")

        # reconstructions: 2 good + one of each defect type
        pick = [i for i, r in enumerate(test) if r["label"] == "good"][:2]
        for t in ["bent", "color", "flip", "scratch"]:
            pick.append(next(i for i, r in enumerate(test) if r["label"] == t))
        diff = ((rec_t - xt) ** 2).mean(1, keepdim=True).cpu()
        diff = diff / diff.max()
        grid([[xt[i].cpu() for i in pick], [rec_t[i].cpu() for i in pick], [diff[i] for i in pick]],
             OUT / "reconstructions.png",
             ["input: " + ", ".join(test[i]["label"] for i in pick), "VAE reconstruction", "squared error"])

        # interpolation between two good training nuts
        za, _ = model.encode(X[0:1].to(dev)); zb, _ = model.encode(X[1:2].to(dev))
        steps = [model.decode((1 - a) * za + a * zb)[0].cpu() for a in np.linspace(0, 1, 8)]
        grid([steps], OUT / "interpolation.png", ["latent interpolation z = (1-a) z_A + a z_B,  a = 0 ... 1"])

        # random samples
        samples = model.decode(torch.randn(8, ZDIM, device=dev)).cpu()
        grid([list(samples)], OUT / "samples.png", ["new nuts decoded from random z ~ N(0, I)"])

        # latent PCA: good vs defect test nuts (PCA via SVD of centred mu)
        M = mu_t.cpu(); Mc = M - M.mean(0)
        U, Sv, Vt = torch.linalg.svd(Mc, full_matrices=False)
        P = (Mc @ Vt[:2].T).numpy()
        plt.figure(figsize=(5, 4.5))
        for lab, col in [("good", "tab:green"), ("bent", "tab:red"), ("color", "tab:orange"),
                         ("flip", "tab:purple"), ("scratch", "tab:blue")]:
            m = np.array([r["label"] == lab for r in test])
            plt.scatter(P[m, 0], P[m, 1], c=col, label=lab, s=30)
        plt.legend(); plt.title("VAE latent space (test set, PCA of mu)"); plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout(); plt.savefig(OUT / "latent_pca.png", dpi=110); plt.close()

        # --- defect synthesizer: real defect regions pasted onto VAE-generated normal nuts
        syn_rows, syn_masks = [], []
        donors = [r for r in S["sup_train"] if r["label"] in ("scratch", "color", "bent")][:6]   # train split only
        base = model.decode(model.encode(X[2:8].to(dev))[0]).cpu()                             # 6 generated normals
        for b, d in zip(base, donors):
            m = load_mask(d["mask"]); src = load(d["path"])
            syn_rows.append(b * (1 - m) + src * m); syn_masks.append(m)
        grid([list(base), syn_rows, syn_masks], OUT / "synthetic_defects.png",
             ["VAE-generated normal nut", "synthetic defect (pasted region from a TRAIN image)", "its label mask"])

    json.dump({"vae_recon_error_test_auroc": auroc, "latent_dim": ZDIM, "beta": BETA, "epochs": epochs,
               "final_loss": hist[-1]}, open(OUT / "metrics.json", "w"), indent=2)
    print("Saved results to", OUT)


if __name__ == "__main__":
    main()