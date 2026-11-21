package dl;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Requirement 3: maths sandbox. Run: java -cp target/classes dl.Demo3Sandbox */
public class Demo3Sandbox {
    static final String OUT = "out/";

    public static void main(String[] args) throws IOException {
        java.nio.file.Files.createDirectories(java.nio.file.Path.of(OUT));
        sandbox();
    }

    static void header(String s) { System.out.println("\n================ " + s + " ================"); }

    static void sandbox() throws IOException {
        header("SANDBOX: tensors, broadcasting, reshape, inner products, SVD, gradient check");
        Tensor a = Tensor.arange(6).reshape(2, 3);
        System.out.println("a = arange(6).reshape(2,3):\n" + a);
        System.out.println("a.reshape(3,-1) shape = " + java.util.Arrays.toString(a.reshape(3, -1).shape));
        Tensor col = new Tensor(new double[]{10, 20}, 2, 1), row = new Tensor(new double[]{1, 2, 3}, 3);
        System.out.println("broadcast (2,3)+(2,1):\n" + a.add(col));
        System.out.println("broadcast (2,3)*(3,):\n" + a.mul(row));
        System.out.println("broadcast (2,1)+(3,) -> outer sum (2,3):\n" + col.add(row));
        try { a.add(new Tensor(new double[]{1, 2}, 2)); }
        catch (IllegalArgumentException e) { System.out.println("expected error: " + e.getMessage()); }
        Tensor u = new Tensor(new double[]{1, 2, 3}, 3), v = new Tensor(new double[]{4, 5, 6}, 3);
        System.out.println("dot(u,v) = " + Tensor.dot(u, v) + "   (1*4+2*5+3*6 = 32)");
        System.out.println("a @ a^T =\n" + a.matmul(a.transpose()));

        // SVD on a small matrix
        Tensor M = Tensor.of2d(new double[][]{{3, 2}, {2, 3}, {2, -2}});
        Linalg.SVD s = Linalg.svd(M);
        System.out.println("SVD of 3x2 M: singular values = " + java.util.Arrays.toString(s.S.data));
        System.out.printf("||M - U S V^T|| = %.2e%n", Linalg.frobDiff(M, s.reconstruct(2)));

        // SVD low-rank approximation of a synthetic 64x64 "surface" image
        int N = 64;
        Tensor img = Tensor.zeros(N, N);
        java.util.Random r = new java.util.Random(1);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double ring = Math.abs(Math.hypot(i - 32, j - 32) - 20) < 3 ? 1 : 0;   // nut-like ring
                img.set(0.5 * Math.sin(j / 6.0) + ring + 0.1 * r.nextGaussian(), i, j);
            }
        Linalg.SVD si = Linalg.svd(img);
        Plot.heatmap(OUT + "sandbox/svd_original.png", img, 4);
        List<String> names = new ArrayList<>(); List<double[]> err = new ArrayList<>();
        double[] errs = new double[N];
        for (int k : new int[]{1, 5, 20}) Plot.heatmap(OUT + "sandbox/svd_rank" + k + ".png", si.reconstruct(k), 4);
        for (int k = 1; k <= N; k++) errs[k - 1] = Linalg.frobDiff(img, si.reconstruct(k)) / img.norm();
        names.add("singular values"); err.add(si.S.data);
        Plot.lines(OUT + "sandbox/singular_values.png", "Singular values of 64x64 image", "index", "sigma", names, err, true);
        Plot.lines(OUT + "sandbox/svd_error.png", "Relative reconstruction error vs rank", "rank k", "error",
                List.of("||A - A_k|| / ||A||"), List.of(errs), false);
        System.out.printf("64x64 image: rank-5 relative error = %.3f, rank-20 = %.3f%n", errs[4], errs[19]);

        // Gradient check on f(x) = sum( sin(x) * x^2 ),  f'(x) = cos(x) x^2 + 2x sin(x)
        Tensor x = Tensor.randn(new java.util.Random(0), 1.0, 5);
        Tensor num = GradCheck.numericalGrad(x, () -> x.map(t -> Math.sin(t) * t * t).sum(), 1e-5);
        Tensor ana = x.map(t -> Math.cos(t) * t * t + 2 * t * Math.sin(t));
        System.out.printf("Gradient check f=sum(sin(x)x^2): max relative error = %.2e%n", GradCheck.maxRelativeError(num, ana));
        // error vs step size h: too large -> truncation error, too small -> floating-point round-off
        double[] errsH = new double[12];
        for (int k = 0; k < 12; k++) {
            double h = Math.pow(10, -1 - k);
            errsH[k] = GradCheck.maxRelativeError(GradCheck.numericalGrad(x, () -> x.map(t -> Math.sin(t) * t * t).sum(), h), ana);
        }
        Plot.lines(OUT + "sandbox/gradcheck_vs_h.png", "Gradient-check error vs h (h = 1e-1 ... 1e-12)", "k (h=10^-(k+1))",
                "rel. error", List.of("max relative error"), List.of(errsH), true);
        System.out.println("Figures written to out/sandbox/");
    }
}