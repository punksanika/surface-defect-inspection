package dl;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/** Requirement 5: optimizers, initialization, generalization. Run: java -cp target/classes dl.Demo5Training */
public class Demo5Training {
    static final String OUT = "out/";

    public static void main(String[] args) throws IOException {
        java.nio.file.Files.createDirectories(java.nio.file.Path.of(OUT));
        optimizers();
        init();
        generalization();
    }

    static void header(String s) { System.out.println("\n================ " + s + " ================"); }

    // ------------------------------------------------------------ optimizers
    static void optimizers() throws IOException {
        header("OPTIMIZERS: Batch GD vs SGD vs Momentum vs Nesterov (two moons, same init)");
        Data train = Data.moons(200, 0.15, 1), val = Data.moons(200, 0.15, 2);
        Object[][] cfg = {
                {"BatchGD", Optimizer.Type.GD, 0.5, 0.0, 200},
                {"SGD_b1", Optimizer.Type.GD, 0.05, 0.0, 1},
                {"Momentum", Optimizer.Type.MOMENTUM, 0.5, 0.9, 200},
                {"Nesterov", Optimizer.Type.NESTEROV, 0.5, 0.9, 200}};
        List<String> names = new ArrayList<>(); List<double[]> curves = new ArrayList<>();
        try (PrintWriter w = new PrintWriter(OUT + "optimizers_summary.csv")) {
            w.println("optimizer,lr,mu,batch,final_train_loss,val_acc");
            for (Object[] c : cfg) {
                MLP net = new MLP(new int[]{2, 16, 1}, MLP.Act.TANH, MLP.Init.XAVIER, 3);   // identical start
                Optimizer opt = new Optimizer((Optimizer.Type) c[1], (double) c[2], (double) c[3]);
                Trainer t = new Trainer().run(net, opt, train, val, 300, (int) c[4], 5, OUT + "optimizers/" + c[0]);
                names.add((String) c[0]); curves.add(t.trainLoss);
                System.out.printf("%-10s %-28s final train loss %.4f  val acc %.3f%n", c[0], opt, t.trainLoss[299], t.finalValAcc);
                w.printf("%s,%s,%s,%s,%.5f,%.4f%n", c[0], c[2], c[3], c[4], t.trainLoss[299], t.finalValAcc);
            }
        }
        Plot.lines(OUT + "optimizers/loss_curves.png", "Optimizer comparison (train loss)", "epoch", "BCE", names, curves, true);
    }

    // ------------------------------------------------------------ initialization
    static void init() throws IOException {
        header("INITIALIZATION: zero vs small normal vs Xavier vs He (deep tanh net)");
        Data train = Data.moons(200, 0.15, 1), val = Data.moons(200, 0.15, 2);
        List<String> names = new ArrayList<>(); List<double[]> curves = new ArrayList<>();
        for (MLP.Init in : MLP.Init.values()) {
            MLP net = new MLP(new int[]{2, 32, 32, 32, 1}, MLP.Act.TANH, in, 3);
            Trainer t = new Trainer().run(net, new Optimizer(Optimizer.Type.MOMENTUM, 0.1, 0.9), train, val, 200, 32, 5,
                    OUT + "init/" + in);
            names.add(in.name()); curves.add(t.trainLoss);
            System.out.printf("%-13s final train loss %.4f  val acc %.3f%n", in, t.trainLoss[199], t.finalValAcc);
        }
        Plot.lines(OUT + "init/loss_curves.png", "Initialization comparison", "epoch", "BCE", names, curves, true);
    }

    // ------------------------------------------------------------ overfitting, L2, augmentation, normalization
    static void generalization() throws IOException {
        header("GENERALIZATION: overfitting, L2, augmentation, input normalization");
        Data small = Data.moons(30, 0.25, 11), val = Data.moons(400, 0.25, 12);
        String[] setups = {"baseline", "L2=0.01", "augment x10"};
        List<String> names = new ArrayList<>(); List<double[]> curves = new ArrayList<>();
        for (String s : setups) {
            MLP net = new MLP(new int[]{2, 64, 64, 1}, MLP.Act.TANH, MLP.Init.XAVIER, 3);
            Data tr = small;
            if (s.startsWith("L2")) net.l2 = 0.01;
            if (s.startsWith("augment")) tr = small.augment(10, 0.1, 4);
            Trainer t = new Trainer().run(net, new Optimizer(Optimizer.Type.MOMENTUM, 0.1, 0.9), tr, val, 400, 16, 5,
                    OUT + "generalization/" + s.replace(' ', '_').replace('=', '_'));
            System.out.printf("%-12s train acc %.3f  val acc %.3f  (gap %.3f)  final train loss %.4f val loss %.4f%n",
                    s, t.finalTrainAcc, t.finalValAcc, t.finalTrainAcc - t.finalValAcc, t.trainLoss[399], t.valLoss[399]);
            names.add(s + " train"); curves.add(t.trainLoss);
            names.add(s + " val"); curves.add(t.valLoss);
            if (s.equals("baseline"))
                Plot.scatter(OUT + "generalization/overfit_boundary.png", "Baseline: 30 points, big net", small.points(), small.labels(), net::predict1, null);
        }
        Plot.lines(OUT + "generalization/train_vs_val.png", "Overfitting: train vs val loss", "epoch", "BCE", names, curves, true);

        // Normalization: features on wildly different scales
        Data raw = Data.moons(200, 0.15, 1), rawVal = Data.moons(200, 0.15, 2);
        Tensor scale = new Tensor(new double[]{100, 0.01}, 1, 2), shift = new Tensor(new double[]{500, -3}, 1, 2);
        Data badTr = new Data(raw.X.mul(scale).add(shift), raw.Y), badVal = new Data(rawVal.X.mul(scale).add(shift), rawVal.Y);
        double[][] st = badTr.stats();   // statistics from TRAIN only (no leakage)
        Data normTr = badTr.standardize(st), normVal = badVal.standardize(st);
        List<String> n2 = new ArrayList<>(); List<double[]> c2 = new ArrayList<>();
        for (int k = 0; k < 2; k++) {
            MLP net = new MLP(new int[]{2, 16, 1}, MLP.Act.TANH, MLP.Init.XAVIER, 3);
            Trainer t = new Trainer().run(net, new Optimizer(Optimizer.Type.MOMENTUM, 0.05, 0.9),
                    k == 0 ? badTr : normTr, k == 0 ? badVal : normVal, 200, 32, 5, OUT + "generalization/" + (k == 0 ? "unnormalized" : "standardized"));
            String nm = k == 0 ? "unnormalized" : "standardized";
            System.out.printf("%-13s val acc %.3f%n", nm, t.finalValAcc);
            n2.add(nm); c2.add(t.trainLoss);
        }
        Plot.lines(OUT + "generalization/normalization.png", "Input standardization", "epoch", "BCE", n2, c2, true);
    }
}