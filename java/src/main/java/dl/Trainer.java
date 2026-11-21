package dl;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/** Training loop + experiment logging (hyperparameters and per-epoch metrics saved to CSV). */
public class Trainer {
    public double[] trainLoss, valLoss;
    public double finalTrainAcc, finalValAcc;

    /**
     * batchSize = N   -> Batch Gradient Descent (one exact gradient per epoch)
     * batchSize = 1   -> Stochastic Gradient Descent (noisy gradient, many updates per epoch)
     */
    public Trainer run(MLP net, Optimizer opt, Data train, Data val, int epochs, int batchSize,
                       long seed, String runDir) throws IOException {
        Random rng = new Random(seed);
        trainLoss = new double[epochs]; valLoss = new double[epochs];
        int n = train.n();
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        for (int e = 0; e < epochs; e++) {
            for (int i = n - 1; i > 0; i--) { int j = rng.nextInt(i + 1); int t = order[i]; order[i] = order[j]; order[j] = t; }
            for (int s = 0; s < n; s += batchSize) {
                Data mb = train.slice(order, s, Math.min(s + batchSize, n));
                net.forward(mb.X);
                net.backward(mb.Y);
                opt.step(net.params(), net.grads());
            }
            trainLoss[e] = net.loss(net.forward(train.X), train.Y);
            valLoss[e] = val == null ? Double.NaN : net.loss(net.forward(val.X), val.Y);
        }
        finalTrainAcc = net.accuracy(train.X, train.Y);
        finalValAcc = val == null ? Double.NaN : net.accuracy(val.X, val.Y);
        if (runDir != null) save(net, opt, epochs, batchSize, seed, runDir);
        return this;
    }

    void save(MLP net, Optimizer opt, int epochs, int batchSize, long seed, String runDir) throws IOException {
        Path d = Path.of(runDir);
        Files.createDirectories(d);
        try (PrintWriter w = new PrintWriter(d.resolve("hparams.txt").toFile())) {
            w.println("layers=" + java.util.Arrays.toString(net.sizes));
            w.println("hidden_activation=" + net.hiddenAct);
            w.println("optimizer=" + opt);
            w.println("epochs=" + epochs + "\nbatch_size=" + batchSize + "\nl2=" + net.l2 + "\nseed=" + seed);
            w.printf("final_train_acc=%.4f%nfinal_val_acc=%.4f%n", finalTrainAcc, finalValAcc);
        }
        try (PrintWriter w = new PrintWriter(d.resolve("metrics.csv").toFile())) {
            w.println("epoch,train_loss,val_loss");
            for (int e = 0; e < trainLoss.length; e++) w.printf("%d,%.6f,%.6f%n", e, trainLoss[e], valLoss[e]);
        }
    }
}