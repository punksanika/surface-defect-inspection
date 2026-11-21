package dl;

import java.io.IOException;
import java.util.List;

/** Requirement 4: neurons, MLP, backprop. Run: java -cp target/classes dl.Demo4Neurons */
public class Demo4Neurons {
    static final String OUT = "out/";

    public static void main(String[] args) throws IOException {
        java.nio.file.Files.createDirectories(java.nio.file.Path.of(OUT));
        neurons();
        mlp();
    }

    static void header(String s) { System.out.println("\n================ " + s + " ================"); }

    static void neurons() throws IOException {
        header("NEURONS: McCulloch-Pitts and sigmoid neuron");
        var AND = Neurons.McCullochPitts.AND(); var OR = Neurons.McCullochPitts.OR(); var NOT = Neurons.McCullochPitts.NOT();
        System.out.println("x1 x2 | AND OR");
        for (int x1 = 0; x1 <= 1; x1++) for (int x2 = 0; x2 <= 1; x2++)
            System.out.printf(" %d  %d |  %d   %d%n", x1, x2, AND.fire(x1, x2), OR.fire(x1, x2));
        System.out.println("NOT(0)=" + NOT.fire(0) + "  NOT(1)=" + NOT.fire(1));
        System.out.println("Grid search for a single MP neuron computing XOR: solutions found = " + Neurons.searchXorSolutions());

        Neurons.SigmoidNeuron sn = new Neurons.SigmoidNeuron(2);
        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; double[] Y = {0, 1, 1, 1};   // OR
        double[] loss = new double[500];
        for (int e = 0; e < 500; e++) loss[e] = sn.trainEpoch(X, Y, 1.0);
        System.out.printf("Sigmoid neuron learning OR: loss %.4f -> %.4f ; outputs: ", loss[0], loss[499]);
        for (double[] x : X) System.out.printf("%.3f ", sn.forward(x));
        System.out.println();
        Plot.lines(OUT + "neurons/sigmoid_or_loss.png", "Sigmoid neuron on OR (BCE loss)", "epoch", "loss",
                List.of("train loss"), List.of(loss), false);
    }

    static void mlp() throws IOException {
        header("MLP: forward, backprop, gradient check, XOR");
        Data d = Data.xor();
        MLP net = new MLP(new int[]{2, 4, 1}, MLP.Act.TANH, MLP.Init.XAVIER, 7);
        net.forward(d.X); net.backward(d.Y);
        double worst = 0;
        for (int k = 0; k < net.params().size(); k++) {
            Tensor p = net.params().get(k), g = net.grads().get(k).copy();
            Tensor num = GradCheck.numericalGrad(p, () -> net.loss(net.forward(d.X), d.Y), 1e-5);
            worst = Math.max(worst, GradCheck.maxRelativeError(num, g));
        }
        System.out.printf("Backprop vs numerical gradient: max relative error = %.2e  (%s)%n", worst, worst < 1e-6 ? "PASS" : "FAIL");
        Trainer t = new Trainer().run(net, new Optimizer(Optimizer.Type.GD, 0.5, 0), d, null, 2000, 4, 1, OUT + "mlp/xor_run");
        System.out.printf("XOR after training: accuracy = %.2f ; outputs: ", t.finalTrainAcc);
        Tensor P = net.forward(d.X);
        for (double p : P.data) System.out.printf("%.3f ", p);
        System.out.println();
        Plot.lines(OUT + "mlp/xor_loss.png", "MLP 2-4-1 on XOR", "epoch", "BCE", List.of("train"), List.of(t.trainLoss), true);
        Plot.scatter(OUT + "mlp/xor_boundary.png", "XOR decision regions", d.points(), d.labels(), net::predict1, null);
    }
}