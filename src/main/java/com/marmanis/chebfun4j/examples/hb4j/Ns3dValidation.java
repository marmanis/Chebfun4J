package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.Locale;

/**
 * Compares the velocity field of two Navier-Stokes-3D checkpoint files
 * and reports per-axis error statistics (max, min, average) of
 * {@code second − first}. The first argument is treated as the baseline.
 *
 * <p>Each file may be either the native binary format written by
 * {@link NavierStokes3D#saveCheckpoint(String)} or the CSV format
 * accepted by {@link NavierStokes3D#loadCheckpoint(String)} (rows of
 * {@code i j k  u v w  wx wy wz} with 1-based indices in {@code [1, N]}).
 * The format is inferred from the file extension: {@code *.csv} → CSV,
 * anything else → native. The two files must share the same cubic grid
 * size N.
 *
 * <p>Native files store the velocity in Fourier space; the loader
 * performs {@code irfft3} to bring them into physical space before the
 * comparison, so a native/CSV mixed pair is compared apples-to-apples.
 *
 * <p>CLI usage:
 * <pre>
 *     java com.marmanis.chebfun4j.examples.hb4j.Ns3dValidation baseline.dat other.csv
 * </pre>
 */
public class Ns3dValidation {
    private static final Logger logger = LoggerFactory.getLogger(Ns3dValidation.class);

    /** Physical-space velocity triple on an {@code N×N×N} cubic grid. */
    public record VelocityField(int n, double[] u, double[] v, double[] w) {}

    /** Per-axis error statistics of a velocity comparison. */
    public record AxisErrors(double max, double min, double avg) {}

    /** Container for the three axis error records. */
    public record Report(AxisErrors x, AxisErrors y, AxisErrors z, int gridSize) {
        @Override public String toString() {
            return String.format(Locale.ROOT,
                "Grid: %dx%dx%d%n" +
                "Max X-axis error: %g;  Min X-axis error: %g;  Avg X-axis error: %g%n" +
                "Max Y-axis error: %g;  Min Y-axis error: %g;  Avg Y-axis error: %g%n" +
                "Max Z-axis error: %g;  Min Z-axis error: %g;  Avg Z-axis error: %g",
                gridSize, gridSize, gridSize,
                x.max, x.min, x.avg,
                y.max, y.min, y.avg,
                z.max, z.min, z.avg);
        }
    }

    /**
     * Loads the velocity triple in physical space from either format.
     * @param path checkpoint file; {@code .csv} extension → CSV loader,
     *             else native binary.
     */
    public static VelocityField load(String path) throws IOException {
        File file = new File(path);
        if (!file.exists()) throw new IOException("File not found: " + path);
        if (path.toLowerCase(Locale.ROOT).endsWith(".csv")) {
            return loadCsv(file);
        }
        return loadNative(file);
    }

    private static VelocityField loadNative(File file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            int n1 = in.readInt();
            int n2 = in.readInt();
            int n3 = in.readInt();
            if (n1 != n2 || n2 != n3) {
                throw new IOException("Native checkpoint " + file + " has non-cubic grid "
                    + n1 + "x" + n2 + "x" + n3 + "; validator requires N=n1=n2=n3");
            }
            int n = n1;
            int half = n / 2 + 1;
            int specSize = n * n * half;
            in.readDouble(); // ttime — unused for velocity comparison

            double[] vxRe = readDoubles(in, specSize);
            double[] vxIm = readDoubles(in, specSize);
            double[] vyRe = readDoubles(in, specSize);
            double[] vyIm = readDoubles(in, specSize);
            double[] vzRe = readDoubles(in, specSize);
            double[] vzIm = readDoubles(in, specSize);

            Shape specShape = new Shape(n, n, half);
            double[] u = Fft.irfft3(new ConcreteNDArray(vxRe, specShape), new ConcreteNDArray(vxIm, specShape)).toDoubleArray();
            double[] v = Fft.irfft3(new ConcreteNDArray(vyRe, specShape), new ConcreteNDArray(vyIm, specShape)).toDoubleArray();
            double[] w = Fft.irfft3(new ConcreteNDArray(vzRe, specShape), new ConcreteNDArray(vzIm, specShape)).toDoubleArray();
            return new VelocityField(n, u, v, w);
        }
    }

    private static double[] readDoubles(DataInputStream in, int count) throws IOException {
        double[] out = new double[count];
        for (int i = 0; i < count; i++) out[i] = in.readDouble();
        return out;
    }

    private static VelocityField loadCsv(File file) throws IOException {
        int detectedMaxIdx = 0;
        int rowCount = 0;
        // First pass: infer N from the max index.
        try (BufferedReader in = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = in.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;
                String[] tok = trimmed.split("[,\\s]+");
                if (tok.length < 9) continue;
                try {
                    int i = Integer.parseInt(tok[0]);
                    int j = Integer.parseInt(tok[1]);
                    int k = Integer.parseInt(tok[2]);
                    detectedMaxIdx = Math.max(detectedMaxIdx, Math.max(i, Math.max(j, k)));
                    rowCount++;
                } catch (NumberFormatException nfe) {
                    // header row — skip
                }
            }
        }
        if (detectedMaxIdx == 0) {
            throw new IOException("CSV " + file + " contained no data rows");
        }
        int n = detectedMaxIdx;
        int expected = n * n * n;
        if (rowCount != expected) {
            throw new IOException("CSV " + file + " expected " + expected + " rows (N=" + n
                + "), got " + rowCount);
        }

        double[] u = new double[expected];
        double[] v = new double[expected];
        double[] w = new double[expected];
        boolean[] filled = new boolean[expected];
        try (BufferedReader in = new BufferedReader(new FileReader(file))) {
            String line;
            int lineNo = 0;
            while ((line = in.readLine()) != null) {
                lineNo++;
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;
                String[] tok = trimmed.split("[,\\s]+");
                if (tok.length < 9) throw new IOException(file + " line " + lineNo
                    + ": expected 9 fields, got " + tok.length);
                int i, j, k;
                try {
                    i = Integer.parseInt(tok[0]);
                    j = Integer.parseInt(tok[1]);
                    k = Integer.parseInt(tok[2]);
                } catch (NumberFormatException nfe) {
                    continue; // header
                }
                if (i < 1 || j < 1 || k < 1 || i > n || j > n || k > n) {
                    throw new IOException(file + " line " + lineNo + ": index ("
                        + i + "," + j + "," + k + ") out of range for N=" + n);
                }
                int idx = ((i - 1) * n + (j - 1)) * n + (k - 1);
                if (filled[idx]) {
                    throw new IOException(file + " line " + lineNo + ": duplicate cell ("
                        + i + "," + j + "," + k + ")");
                }
                u[idx] = Double.parseDouble(tok[3]);
                v[idx] = Double.parseDouble(tok[4]);
                w[idx] = Double.parseDouble(tok[5]);
                filled[idx] = true;
            }
        }
        return new VelocityField(n, u, v, w);
    }

    /**
     * Compares two velocity fields and reports per-axis error stats.
     * Errors are computed as {@code second − first}.
     */
    public static Report compare(VelocityField baseline, VelocityField other) {
        if (baseline.n != other.n) {
            throw new IllegalArgumentException("Grid size mismatch: baseline N=" + baseline.n
                + ", other N=" + other.n);
        }
        return new Report(
            axisStats(baseline.u, other.u),
            axisStats(baseline.v, other.v),
            axisStats(baseline.w, other.w),
            baseline.n);
    }

    private static AxisErrors axisStats(double[] a, double[] b) {
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        double sum = 0.0;
        int n = a.length;
        for (int i = 0; i < n; i++) {
            double e = b[i] - a[i];
            if (e > max) max = e;
            if (e < min) min = e;
            sum += e;
        }
        return new AxisErrors(max, min, sum / n);
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("usage: Ns3dValidation <baseline> <other>");
            System.err.println("  Each file may be a native checkpoint (.dat) or CSV (.csv).");
            System.exit(2);
        }
        VelocityField baseline = load(args[0]);
        VelocityField other    = load(args[1]);
        Report report = compare(baseline, other);
        logger.info("Baseline: {}", args[0]);
        logger.info("Other   : {}", args[1]);
        System.out.println(report);
    }
}
