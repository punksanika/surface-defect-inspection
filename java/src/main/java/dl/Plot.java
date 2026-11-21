package dl;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/** Tiny plotting helpers that write PNG files using only the Java standard library. */
public class Plot {
    static final Color[] COLORS = {new Color(31, 119, 180), new Color(255, 127, 14), new Color(44, 160, 44),
            new Color(214, 39, 40), new Color(148, 103, 189), new Color(140, 86, 75)};

    /** Line plot of several series (e.g. loss curves). */
    public static void lines(String file, String title, String xLabel, String yLabel,
                             List<String> names, List<double[]> series, boolean logY) throws IOException {
        int W = 800, H = 500, L = 70, R = 170, T = 40, B = 50;
        double ymin = Double.MAX_VALUE, ymax = -Double.MAX_VALUE; int n = 0;
        for (double[] s : series) {
            n = Math.max(n, s.length);
            for (double v : s) { double y = logY ? Math.log10(Math.max(v, 1e-12)) : v; ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); }
        }
        if (ymax - ymin < 1e-12) ymax = ymin + 1;
        BufferedImage img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);
        g.setColor(Color.BLACK);
        g.drawString(title, L, 25);
        g.drawRect(L, T, W - L - R, H - T - B);
        g.drawString(xLabel, (W - R) / 2, H - 15);
        g.drawString(yLabel + (logY ? " (log10)" : ""), 5, T - 5);
        for (int k = 0; k <= 4; k++) {
            double y = ymin + k * (ymax - ymin) / 4;
            int py = H - B - (int) (k * (H - T - B) / 4.0);
            g.drawString(String.format("%.2f", y), 5, py + 4);
        }
        g.drawString("0", L, H - B + 15);
        g.drawString(String.valueOf(n - 1), W - R - 20, H - B + 15);
        g.setStroke(new BasicStroke(2f));
        for (int s = 0; s < series.size(); s++) {
            double[] d = series.get(s);
            g.setColor(COLORS[s % COLORS.length]);
            int px0 = -1, py0 = -1;
            for (int i = 0; i < d.length; i++) {
                double y = logY ? Math.log10(Math.max(d[i], 1e-12)) : d[i];
                int px = L + (int) ((W - L - R) * i / (double) Math.max(n - 1, 1));
                int py = H - B - (int) ((H - T - B) * (y - ymin) / (ymax - ymin));
                if (px0 >= 0) g.drawLine(px0, py0, px, py);
                px0 = px; py0 = py;
            }
            g.drawString(names.get(s), W - R + 10, T + 20 + 20 * s);
        }
        save(img, file);
    }

    /** Scatter of 2-D points coloured by class, with an optional background showing classifier output in [0,1]. */
    public static void scatter(String file, String title, double[][] pts, int[] cls,
                               java.util.function.BiFunction<Double, Double, Double> field,
                               double[][] centers) throws IOException {
        int S = 500;
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double[] p : pts) for (double v : p) { min = Math.min(min, v); max = Math.max(max, v); }
        double pad = 0.1 * (max - min); min -= pad; max += pad;
        final double mn = min, mx = max;
        BufferedImage img = new BufferedImage(S, S, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = setup(img);
        if (field != null)
            for (int px = 0; px < S; px += 4)
                for (int py = 0; py < S; py += 4) {
                    double x = mn + (mx - mn) * px / S, y = mx - (mx - mn) * py / S;
                    double v = Math.max(0, Math.min(1, field.apply(x, y)));
                    g.setColor(new Color((int) (255 - 60 * v), (int) (235 - 20 * Math.abs(v - .5)), (int) (195 + 60 * v)));
                    g.fillRect(px, py, 4, 4);
                }
        for (int i = 0; i < pts.length; i++) {
            int px = (int) ((pts[i][0] - mn) / (mx - mn) * S), py = (int) ((mx - pts[i][1]) / (mx - mn) * S);
            g.setColor(COLORS[cls[i] % COLORS.length]);
            g.fillOval(px - 3, py - 3, 7, 7);
        }
        if (centers != null) {
            g.setColor(Color.BLACK); g.setStroke(new BasicStroke(3f));
            for (double[] c : centers) {
                int px = (int) ((c[0] - mn) / (mx - mn) * S), py = (int) ((mx - c[1]) / (mx - mn) * S);
                g.drawLine(px - 8, py - 8, px + 8, py + 8); g.drawLine(px - 8, py + 8, px + 8, py - 8);
            }
        }
        g.setColor(Color.BLACK); g.drawString(title, 10, 20);
        save(img, file);
    }

    /** Grayscale image of a 2-D tensor (e.g. matrix, SVD reconstruction, Hopfield pattern). */
    public static void heatmap(String file, Tensor m, int cell) throws IOException {
        int r = m.shape[0], c = m.shape[1];
        double min = Double.MAX_VALUE, max = -Double.MAX_VALUE;
        for (double v : m.data) { min = Math.min(min, v); max = Math.max(max, v); }
        BufferedImage img = new BufferedImage(c * cell, r * cell, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++) {
                int v = (int) (255 * (m.get(i, j) - min) / Math.max(max - min, 1e-12));
                g.setColor(new Color(v, v, v));
                g.fillRect(j * cell, i * cell, cell, cell);
            }
        save(img, file);
    }

    static Graphics2D setup(BufferedImage img) {
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE); g.fillRect(0, 0, img.getWidth(), img.getHeight());
        return g;
    }

    static void save(BufferedImage img, String file) throws IOException {
        File f = new File(file);
        if (f.getParentFile() != null) f.getParentFile().mkdirs();
        ImageIO.write(img, "png", f);
    }
}