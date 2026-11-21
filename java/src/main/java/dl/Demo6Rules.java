package dl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Requirement 6: learning rules + Hopfield energy model. Run: java -cp target/classes dl.Demo6Rules */
public class Demo6Rules {
    static final String OUT = "out/";

    public static void main(String[] args) throws IOException {
        java.nio.file.Files.createDirectories(java.nio.file.Path.of(OUT));
        rules();
        hopfield();
    }

    static void header(String s) { System.out.println("\n================ " + s + " ================"); }

    static void rules() throws IOException {
        header("LEARNING RULES: Hebbian, Oja, competitive, error-correction");
        double[][] X = LearningRules.correlatedData(500, 30, 1);
        double[] pc = LearningRules.pcaDirection(X);
        double[] wh = {0.1, 0.0}, wo = {0.1, 0.0};
        double[] nh = LearningRules.hebbian(X, wh, 1e-4, 30);
        double[] no = LearningRules.oja(X, wo, 1e-3, 30);
        System.out.printf("PC1 from our SVD: (%.3f, %.3f)%n", pc[0], pc[1]);
        System.out.printf("Hebbian: ||w|| grew from 0.1 to %.3e ; |cos(w, PC1)| = %.4f%n", nh[29], LearningRules.absCosine(wh, pc));
        System.out.printf("Oja    : ||w|| = %.4f (-> 1)        ; |cos(w, PC1)| = %.4f%n", no[29], LearningRules.absCosine(wo, pc));
        Plot.lines(OUT + "rules/weight_norms.png", "Weight norm: Hebbian explodes, Oja -> 1", "epoch", "||w||",
                List.of("Hebbian", "Oja"), List.of(nh, no), true);

        double[][] centers = {{0, 0}, {4, 1}, {1, 4}};
        int[] lab = new int[300];
        double[][] B = LearningRules.blobs(100, centers, 0.6, 2, lab);
        double[][] protos = LearningRules.competitive(B, 3, 0.05, 20, 3);
        System.out.print("Competitive learning prototypes: ");
        for (double[] p : protos) System.out.printf("(%.2f, %.2f) ", p[0], p[1]);
        System.out.println(" true centres (0,0) (4,1) (1,4)");
        Plot.scatter(OUT + "rules/competitive.png", "Winner-take-all prototypes (X)", B, lab, null, protos);

        int[] t = new int[300];
        for (int i = 0; i < 300; i++) t[i] = lab[i] == 0 ? 0 : 1;   // class 0 vs rest: linearly separable
        double[] w = {0, 0, 0};
        int[] mistakes = LearningRules.perceptron(B, t, w, 0.1, 15);
        System.out.println("Perceptron (error-correction) mistakes per epoch: " + java.util.Arrays.toString(mistakes));
        double[] m = new double[mistakes.length];
        for (int i = 0; i < m.length; i++) m[i] = mistakes[i];
        Plot.lines(OUT + "rules/perceptron_mistakes.png", "Error-correction learning", "epoch", "mistakes",
                List.of("mistakes"), List.of(m), false);
    }

    static void hopfield() throws IOException {
        header("HOPFIELD: energy-based associative memory");
        int[][] P = {
                Hopfield.fromAscii("..######..", ".#......#.", "#........#", "#..####..#", "#.#....#.#",
                        "#.#....#.#", "#..####..#", "#........#", ".#......#.", "..######.."),   // nut (ring)
                Hopfield.fromAscii("##########", "#........#", "#........#", "#..####..#", "#..#..#..#",
                        "#..#..#..#", "#..####..#", "#........#", "#........#", "##########"),   // square nut
                Hopfield.fromAscii("....##....", "....##....", "....##....", "##########", "##########",
                        "....##....", "....##....", "....##....", "....##....", "....##....")};  // cross
        Hopfield h = new Hopfield(100);
        h.store(P);
        List<String> names = new ArrayList<>(); List<double[]> traces = new ArrayList<>();
        for (int k = 0; k < P.length; k++) {
            int[] s = Hopfield.corrupt(P[k], 0.2, 10 + k);
            int before = Hopfield.hamming(s, P[k]);
            Plot.heatmap(OUT + "hopfield/p" + k + "_stored.png", Hopfield.toImage(P[k], 10), 20);
            Plot.heatmap(OUT + "hopfield/p" + k + "_corrupted.png", Hopfield.toImage(s, 10), 20);
            double[] tr = h.recall(s, 5, 20 + k);
            Plot.heatmap(OUT + "hopfield/p" + k + "_recalled.png", Hopfield.toImage(s, 10), 20);
            System.out.printf("Pattern %d: wrong pixels %d -> %d ; energy %.2f -> %.2f%n",
                    k, before, Hopfield.hamming(s, P[k]), tr[0], tr[tr.length - 1]);
            names.add("pattern " + k); traces.add(tr);
        }
        Plot.lines(OUT + "hopfield/energy.png", "Energy never increases during recall", "update step", "E",
                names, traces, false);
    }
}