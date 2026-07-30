package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Ns3dValidationTest {

    private static final int N = 8;

    // --- direct compare() with hand-crafted fields ---------------------------

    @Test
    void compareIdenticalFieldsGivesZeroError() {
        Ns3dValidation.VelocityField a = makeSmoothField(N, 0.0);
        Ns3dValidation.VelocityField b = makeSmoothField(N, 0.0);
        Ns3dValidation.Report r = Ns3dValidation.compare(a, b);
        assertEquals(0.0, r.x().max());
        assertEquals(0.0, r.x().min());
        assertEquals(0.0, r.x().avg());
        assertEquals(0.0, r.y().max());
        assertEquals(0.0, r.z().max());
    }

    @Test
    void compareWithKnownConstantOffset() {
        // other = baseline + 0.25 on every component ⇒ max = min = avg = 0.25.
        Ns3dValidation.VelocityField baseline = makeSmoothField(N, 0.0);
        Ns3dValidation.VelocityField other    = makeSmoothField(N, 0.25);
        Ns3dValidation.Report r = Ns3dValidation.compare(baseline, other);
        assertEquals(0.25, r.x().max(), 1e-14);
        assertEquals(0.25, r.x().min(), 1e-14);
        assertEquals(0.25, r.x().avg(), 1e-14);
        assertEquals(0.25, r.y().avg(), 1e-14);
        assertEquals(0.25, r.z().avg(), 1e-14);
    }

    @Test
    void compareRejectsGridSizeMismatch() {
        Ns3dValidation.VelocityField a = makeSmoothField(N, 0.0);
        Ns3dValidation.VelocityField b = makeSmoothField(N + 2, 0.0);
        assertThrows(IllegalArgumentException.class, () -> Ns3dValidation.compare(a, b));
    }

    @Test
    void reportToStringMentionsAllThreeAxes() {
        Ns3dValidation.Report r = Ns3dValidation.compare(
            makeSmoothField(N, 0.0), makeSmoothField(N, 1e-9));
        String s = r.toString();
        assertTrue(s.contains("X-axis"));
        assertTrue(s.contains("Y-axis"));
        assertTrue(s.contains("Z-axis"));
        assertTrue(s.contains(N + "x" + N + "x" + N));
    }

    // --- round-trip through CSV ---------------------------------------------

    @Test
    void csvRoundTripPreservesField(@TempDir Path tmp) throws IOException {
        Ns3dValidation.VelocityField expected = makeSmoothField(N, 0.0);
        Path csv = tmp.resolve("field.csv");
        writeCsv(csv, expected);

        Ns3dValidation.VelocityField loaded = Ns3dValidation.load(csv.toString());
        assertEquals(expected.n(), loaded.n());
        Ns3dValidation.Report r = Ns3dValidation.compare(expected, loaded);
        assertEquals(0.0, Math.max(Math.abs(r.x().max()), Math.abs(r.x().min())), 1e-14);
        assertEquals(0.0, Math.max(Math.abs(r.y().max()), Math.abs(r.y().min())), 1e-14);
        assertEquals(0.0, Math.max(Math.abs(r.z().max()), Math.abs(r.z().min())), 1e-14);
    }

    // --- round-trip through native binary + mixed CSV/native ----------------

    @Test
    void nativeCsvMixedComparisonIsNearZero(@TempDir Path tmp) throws IOException {
        Ns3dValidation.VelocityField field = makeSmoothField(N, 0.0);

        Path csv = tmp.resolve("field.csv");
        writeCsv(csv, field);

        Path bin = tmp.resolve("field.dat");
        writeNative(bin, field);

        // Both loaders should recover the same field to FFT precision.
        Ns3dValidation.VelocityField fromCsv    = Ns3dValidation.load(csv.toString());
        Ns3dValidation.VelocityField fromNative = Ns3dValidation.load(bin.toString());

        Ns3dValidation.Report r = Ns3dValidation.compare(fromCsv, fromNative);
        double worst = maxAbs(r);
        assertTrue(worst < 1e-10,
            "expected CSV vs native round-trip to agree; worst=" + worst);
    }

    // --- helpers -------------------------------------------------------------

    /** Analytic velocity field, plus an optional constant per-component offset. */
    private static Ns3dValidation.VelocityField makeSmoothField(int n, double offset) {
        double[] u = new double[n * n * n];
        double[] v = new double[n * n * n];
        double[] w = new double[n * n * n];
        double dx = 2.0 * Math.PI / n;
        for (int i = 0; i < n; i++) {
            double x = i * dx;
            for (int j = 0; j < n; j++) {
                double y = j * dx;
                for (int k = 0; k < n; k++) {
                    double z = k * dx;
                    int idx = (i * n + j) * n + k;
                    u[idx] = Math.sin(x) * Math.cos(y)             + offset;
                    v[idx] = Math.cos(x) * Math.sin(z)             + offset;
                    w[idx] = Math.sin(y) * Math.cos(z) * 0.5       + offset;
                }
            }
        }
        return new Ns3dValidation.VelocityField(n, u, v, w);
    }

    private static void writeCsv(Path file, Ns3dValidation.VelocityField f) throws IOException {
        int n = f.n();
        double[] u = f.u(), v = f.v(), w = f.w();
        try (BufferedWriter out = Files.newBufferedWriter(file)) {
            out.write("# i j k  u v w  wx wy wz\n");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        int idx = (i * n + j) * n + k;
                        // Vorticity columns are unused by the validator, write zeros.
                        out.write(String.format(Locale.ROOT,
                            "%d %d %d  %.17g %.17g %.17g  0 0 0%n",
                            i + 1, j + 1, k + 1, u[idx], v[idx], w[idx]));
                    }
                }
            }
        }
    }

    /** Writes the same binary layout produced by NavierStokes3D.saveCheckpoint. */
    private static void writeNative(Path file, Ns3dValidation.VelocityField f) throws IOException {
        int n = f.n();
        Shape realShape = new Shape(n, n, n);
        NDArray[] uSpec = Fft.rfft3(new ConcreteNDArray(f.u(), realShape));
        NDArray[] vSpec = Fft.rfft3(new ConcreteNDArray(f.v(), realShape));
        NDArray[] wSpec = Fft.rfft3(new ConcreteNDArray(f.w(), realShape));

        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(file.toFile()))) {
            out.writeInt(n);
            out.writeInt(n);
            out.writeInt(n);
            out.writeDouble(0.0); // ttime
            writeAll(out, uSpec[0].toDoubleArray());
            writeAll(out, uSpec[1].toDoubleArray());
            writeAll(out, vSpec[0].toDoubleArray());
            writeAll(out, vSpec[1].toDoubleArray());
            writeAll(out, wSpec[0].toDoubleArray());
            writeAll(out, wSpec[1].toDoubleArray());
        }
    }

    private static void writeAll(DataOutputStream out, double[] a) throws IOException {
        for (double d : a) out.writeDouble(d);
    }

    private static double maxAbs(Ns3dValidation.Report r) {
        double m = 0;
        m = Math.max(m, Math.max(Math.abs(r.x().max()), Math.abs(r.x().min())));
        m = Math.max(m, Math.max(Math.abs(r.y().max()), Math.abs(r.y().min())));
        m = Math.max(m, Math.max(Math.abs(r.z().max()), Math.abs(r.z().min())));
        return m;
    }
}
