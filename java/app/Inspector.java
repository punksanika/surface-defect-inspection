import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Inspector: the deployed inspection application in Java (Requirement 10).
 *
 *   image -> resize 224 + ImageNet normalisation (Java)
 *         -> ResNet-18 feature extractor (ONNX Runtime)        (1,384,28,28)
 *         -> 784 patch vectors, nearest neighbour in memory bank (Java, parallel)
 *         -> image score = max patch distance -> PASS / REJECT (threshold from validation set)
 *
 * Usage (from the project folder):
 *   java -cp target/app:lib/onnxruntime-1.23.0.jar Inspector inspect <image.png>
 *   java -cp target/app:lib/onnxruntime-1.23.0.jar Inspector evaluate
 *
 * Optional last argument = model folder (default "deploy"), e.g. the noise-robust model:
 *   java -cp target/app:lib/onnxruntime-1.23.0.jar Inspector evaluate deploy_robust
 *   (results go to experiments/java_app_robust/ so the original results are kept)
 * Any input blur is inside the .onnx file, so this Java code does not need to know about it.
 */
public class Inspector {
    final int size, C, H, W, M;
    final double threshold;
    final float[] mean = new float[3], std = new float[3];
    final float[] bank;          // M*C, row-major
    final float[] bankNorm2;     // ||b||^2 per bank row (precomputed)
    final OrtEnvironment env;
    final OrtSession session;

    Inspector(Path deploy) throws Exception {
        Properties p = new Properties();
        try (FileInputStream in = new FileInputStream(deploy.resolve("model.properties").toFile())) { p.load(in); }
        size = Integer.parseInt(p.getProperty("image_size"));
        C = Integer.parseInt(p.getProperty("feature_dim"));
        H = Integer.parseInt(p.getProperty("feature_h"));
        W = Integer.parseInt(p.getProperty("feature_w"));
        M = Integer.parseInt(p.getProperty("bank_size"));
        threshold = Double.parseDouble(p.getProperty("image_threshold"));
        String[] ms = p.getProperty("mean").split(","), ss = p.getProperty("std").split(",");
        for (int i = 0; i < 3; i++) { mean[i] = Float.parseFloat(ms[i]); std[i] = Float.parseFloat(ss[i]); }

        byte[] raw = Files.readAllBytes(deploy.resolve("memory_bank.f32"));
        if (raw.length != 4L * M * C) throw new IllegalStateException("memory bank size mismatch");
        bank = new float[M * C];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(bank);
        bankNorm2 = new float[M];
        for (int m = 0; m < M; m++) {
            float s = 0;
            for (int c = 0; c < C; c++) { float v = bank[m * C + c]; s += v * v; }
            bankNorm2[m] = s;
        }
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(deploy.resolve("patchcore_r18.onnx").toString(), new OrtSession.SessionOptions());
    }

    /** Resize with area averaging (anti-aliased, close to PIL) and return RGB values in [0,1], CHW layout. */
    float[] toCHW(BufferedImage src) {
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.drawImage(src.getScaledInstance(size, size, Image.SCALE_AREA_AVERAGING), 0, 0, null);
        g.dispose();
        float[] x = new float[3 * size * size];
        for (int y = 0; y < size; y++)
            for (int xx = 0; xx < size; xx++) {
                int rgb = img.getRGB(xx, y);
                x[y * size + xx] = ((rgb >> 16) & 255) / 255f;
                x[size * size + y * size + xx] = ((rgb >> 8) & 255) / 255f;
                x[2 * size * size + y * size + xx] = (rgb & 255) / 255f;
            }
        return x;
    }

    /** Result of one inspection. patchScores is H*W (28x28). */
    record Result(double score, boolean reject, float[] patchScores, double ms) {}

    Result inspect(float[] chw01) throws Exception {
        long t0 = System.nanoTime();
        float[] x = new float[chw01.length];
        int hw = size * size;
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < hw; i++) x[c * hw + i] = (chw01[c * hw + i] - mean[c]) / std[c];
        float[] feat;     // C*H*W
        try (OnnxTensor t = OnnxTensor.createTensor(env, FloatBuffer.wrap(x), new long[]{1, 3, size, size});
             OrtSession.Result r = session.run(Map.of("image", t))) {
            FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
            feat = new float[fb.remaining()];
            fb.get(feat);
        }
        int P = H * W;
        float[] scores = new float[P];
        // nearest neighbour for each patch:  ||p - b||^2 = ||p||^2 + ||b||^2 - 2 p.b   (parallel over patches)
        IntStream.range(0, P).parallel().forEach(pi -> {
            float[] v = new float[C];
            float pn = 0;
            for (int c = 0; c < C; c++) { v[c] = feat[c * P + pi]; pn += v[c] * v[c]; }
            float best = Float.MAX_VALUE;
            for (int m = 0; m < M; m++) {
                float dot = 0;
                int off = m * C;
                for (int c = 0; c < C; c++) dot += v[c] * bank[off + c];
                float d2 = pn + bankNorm2[m] - 2 * dot;
                if (d2 < best) best = d2;
            }
            scores[pi] = (float) Math.sqrt(Math.max(best, 0));
        });
        double s = 0;
        for (float f : scores) s = Math.max(s, f);
        return new Result(s, s >= threshold, scores, (System.nanoTime() - t0) / 1e6);
    }

    void saveHeatmap(BufferedImage src, float[] patch, double vmax, Path out) throws IOException {
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.drawImage(src.getScaledInstance(size, size, Image.SCALE_AREA_AVERAGING), 0, 0, null);
        g.dispose();
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                double v = Math.min(1, patch[(y * H / size) * W + (x * W / size)] / vmax);
                int rgb = img.getRGB(x, y);
                int r = (int) (0.55 * ((rgb >> 16) & 255) + 0.45 * 255 * v);
                int gg = (int) (0.55 * ((rgb >> 8) & 255) + 0.45 * 255 * (1 - Math.abs(2 * v - 1)));
                int b = (int) (0.55 * (rgb & 255) + 0.45 * 255 * (1 - v));
                img.setRGB(x, y, (r << 16) | (gg << 8) | b);
            }
        ImageIO.write(img, "png", out.toFile());
    }

    // ------------------------------------------------------------ evaluation helpers
    static double auroc(List<Double> s, List<Integer> y) {
        double pairs = 0, good = 0;
        for (int i = 0; i < s.size(); i++)
            for (int j = 0; j < s.size(); j++)
                if (y.get(i) == 1 && y.get(j) == 0) {
                    pairs++;
                    if (s.get(i) > s.get(j)) good++; else if (s.get(i).equals(s.get(j))) good += 0.5;
                }
        return good / pairs;
    }

    static float[] perturb(float[] x, String kind, Random rng) {
        float[] o = x.clone();
        for (int i = 0; i < o.length; i++) {
            float v = o[i];
            switch (kind) {
                case "brighter_20pct" -> v *= 1.2f;
                case "darker_20pct" -> v *= 0.8f;
                case "noise_sigma_0.03" -> v += (float) (0.03 * rng.nextGaussian());
                default -> { }
            }
            o[i] = Math.max(0f, Math.min(1f, v));
        }
        return o;
    }

    void evaluate(Path deploy, Path outDir) throws Exception {
        Files.createDirectories(outDir);
        List<String> lines = Files.readAllLines(deploy.resolve("test_list.csv"));
        List<String> paths = new ArrayList<>(), labels = new ArrayList<>();
        List<Double> pyScores = new ArrayList<>();
        for (String l : lines.subList(1, lines.size())) {
            String[] f = l.split(",");
            paths.add(f[0]); labels.add(f[1]); pyScores.add(Double.parseDouble(f[2]));
        }
        List<float[]> imgs = new ArrayList<>();
        for (String p : paths) imgs.add(toCHW(ImageIO.read(Path.of(p).toFile())));
        inspect(imgs.get(0));   // warm-up

        String[] conditions = {"clean", "brighter_20pct", "darker_20pct", "noise_sigma_0.03"};
        try (PrintWriter w = new PrintWriter(outDir.resolve("java_eval.csv").toFile())) {
            w.println("condition,path,label,java_score,python_score,decision");
            for (String cond : conditions) {
                Random rng = new Random(0);
                List<Double> sc = new ArrayList<>(); List<Integer> y = new ArrayList<>();
                int tp = 0, fp = 0, fn = 0, tn = 0, agree = 0; double ms = 0, maxDiff = 0;
                for (int i = 0; i < imgs.size(); i++) {
                    Result r = inspect(perturb(imgs.get(i), cond, rng));
                    int truth = labels.get(i).equals("good") ? 0 : 1;
                    sc.add(r.score()); y.add(truth); ms += r.ms();
                    if (r.reject() && truth == 1) tp++; else if (r.reject()) fp++; else if (truth == 1) fn++; else tn++;
                    if (cond.equals("clean")) {
                        maxDiff = Math.max(maxDiff, Math.abs(r.score() - pyScores.get(i)));
                        if (r.reject() == (pyScores.get(i) >= threshold)) agree++;
                    }
                    w.printf("%s,%s,%s,%.5f,%.5f,%s%n", cond, paths.get(i), labels.get(i), r.score(), pyScores.get(i),
                            r.reject() ? "REJECT" : "PASS");
                }
                double rec = tp / (double) Math.max(tp + fn, 1), prec = tp / (double) Math.max(tp + fp, 1);
                System.out.printf("%n[%s]  AUROC %.3f | TP %d FP %d FN %d TN %d | catch %.1f%%  FRR %.1f%%  F1 %.3f | %.1f ms/img (%.1f FPS)%n",
                        cond, auroc(sc, y), tp, fp, fn, tn, 100 * rec, 100 * fp / (double) Math.max(fp + tn, 1),
                        2 * prec * rec / Math.max(prec + rec, 1e-9), ms / imgs.size(), 1000 * imgs.size() / ms);
                if (cond.equals("clean"))
                    System.out.printf("  parity with Python: decisions agree on %d/%d images, max |score difference| = %.4f%n",
                            agree, imgs.size(), maxDiff);
            }
        }
        Runtime rt = Runtime.getRuntime();
        System.out.printf("%nJava heap in use: %.0f MB (memory bank alone: %.1f MB)%n",
                (rt.totalMemory() - rt.freeMemory()) / 1e6, 4.0 * M * C / 1e6);
        System.out.println("Per-image results: " + outDir.resolve("java_eval.csv"));
    }

    public static void main(String[] args) throws Exception {
        String mode = args.length > 0 ? args[0] : "evaluate";
        // model folder: "deploy" by default; inspect takes it as 3rd argument, evaluate as 2nd
        int dirArg = mode.equals("inspect") ? 2 : 1;
        Path deploy = Path.of(args.length > dirArg ? args[dirArg] : "deploy");
        Inspector ins = new Inspector(deploy);
        System.out.printf("Loaded ONNX model + memory bank (%d x %d), threshold %.4f%n", ins.M, ins.C, ins.threshold);
        if (mode.equals("inspect")) {
            Path img = Path.of(args[1]);
            BufferedImage src = ImageIO.read(img.toFile());
            ins.inspect(ins.toCHW(src));                         // warm-up
            Result r = ins.inspect(ins.toCHW(src));
            System.out.printf("%s -> %s  (score %.3f, threshold %.3f, %.1f ms)%n", img.getFileName(),
                    r.reject() ? "REJECT" : "PASS", r.score(), ins.threshold, r.ms());
            Path out = deploy.resolve("heatmap_" + img.getFileName());
            ins.saveHeatmap(src, r.patchScores(), Math.max(r.score(), ins.threshold * 1.5), out);
            System.out.println("Heatmap saved to " + out);
        } else {
            String outName = deploy.getFileName().toString().equals("deploy") ? "java_app"
                    : "java_app" + deploy.getFileName().toString().replace("deploy", "");   // deploy_robust -> java_app_robust
            ins.evaluate(deploy, Path.of("experiments", outName));
        }
    }
}