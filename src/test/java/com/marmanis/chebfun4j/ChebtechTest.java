package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ChebTransform;
import com.marmanis.chebfun4j.util.ChebyshevPoints;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Correctness of the Chebtech core: value ↔ coefficient round-trips,
 * Clenshaw evaluation matching values at the grid, and known polynomial
 * coefficients coming out exact.
 */
public class ChebtechTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSecondKindGridEndpoints() {
        double[] g = ChebyshevPoints.secondKind(17);
        assertEquals(1.0, g[0], 0.0, "x_0");
        assertEquals(-1.0, g[16], 0.0, "x_{n-1}");
        assertClose(Math.cos(Math.PI / 16), g[1], 1e-15, "x_1");
    }

    @Test
    public void testValsCoeffsRoundTrip() {
        // Random values, transform there and back, expect identity.
        int n = 65; // 2^6 + 1
        double[] v = new double[n];
        java.util.Random rng = new java.util.Random(5);
        for (int i = 0; i < n; i++) v[i] = rng.nextGaussian();
        double[] c = ChebTransform.vals2coeffs(v);
        double[] vBack = ChebTransform.coeffs2vals(c);
        for (int i = 0; i < n; i++) assertClose(v[i], vBack[i], 1e-10, "round-trip[" + i + "]");
    }

    @Test
    public void testConstantFunctionHasOnlyC0() {
        // f(x) = 3.7 on grid.
        int n = 9;
        double[] v = new double[n];
        java.util.Arrays.fill(v, 3.7);
        double[] c = ChebTransform.vals2coeffs(v);
        assertClose(3.7, c[0], 1e-13, "c_0");
        for (int k = 1; k < n; k++) assertClose(0.0, c[k], 1e-13, "c_" + k);
    }

    @Test
    public void testLinearFunctionHasOnlyC1() {
        // f(x) = x on Chebyshev grid: T_1(x) = x, so c_1 = 1, all others 0.
        int n = 17;
        double[] xs = ChebyshevPoints.secondKind(n);
        double[] c = ChebTransform.vals2coeffs(xs);
        assertClose(0.0, c[0], 1e-13, "c_0");
        assertClose(1.0, c[1], 1e-13, "c_1");
        for (int k = 2; k < n; k++) assertClose(0.0, c[k], 1e-13, "c_" + k);
    }

    @Test
    public void testQuadraticHasOnlyC0AndC2() {
        // f(x) = 2x^2 - 1 = T_2(x), so c_2 = 1, else 0.
        int n = 17;
        double[] xs = ChebyshevPoints.secondKind(n);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = 2 * xs[i] * xs[i] - 1;
        double[] c = ChebTransform.vals2coeffs(v);
        for (int k = 0; k < n; k++) {
            double want = (k == 2) ? 1.0 : 0.0;
            assertClose(want, c[k], 1e-13, "c_" + k);
        }
    }

    @Test
    public void testChebtechEvalMatchesValuesAtGrid() {
        int n = 33;
        double[] xs = ChebyshevPoints.secondKind(n);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = Math.exp(xs[i]) * Math.sin(3 * xs[i]);
        Chebtech t = Chebtech.fromValues(v);
        double[] back = t.values();
        for (int i = 0; i < n; i++) assertClose(v[i], back[i], 1e-12, "values@grid[" + i + "]");
        for (int i = 0; i < n; i++) {
            double ev = t.eval(xs[i]);
            assertClose(v[i], ev, 1e-12, "clenshaw@grid[" + i + "]");
        }
    }

    @Test
    public void testChebtechEvalMatchesFunctionInside() {
        // Higher-degree resolution: pick n = 65, compare interior points.
        int n = 65;
        double[] xs = ChebyshevPoints.secondKind(n);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = Math.exp(xs[i]);
        Chebtech t = Chebtech.fromValues(v);
        double[] test = {0.0, 0.25, -0.5, 0.9, -0.9, 0.123456};
        for (double x : test) {
            assertClose(Math.exp(x), t.eval(x), 1e-12, "eval@" + x);
        }
    }

    @Test
    public void testProlongIsFunctionPreserving() {
        int n = 17;
        double[] xs = ChebyshevPoints.secondKind(n);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) v[i] = xs[i] * xs[i] * xs[i]; // x^3
        Chebtech small = Chebtech.fromValues(v);
        Chebtech big = small.prolong(33);
        assertEquals(33, big.length());
        double[] test = {0.0, 0.3, -0.7, 0.9};
        for (double x : test) {
            assertClose(small.eval(x), big.eval(x), 1e-13, "prolong preserves eval @" + x);
        }
    }

    @Test
    public void testSimplifyDropsTrailingNoise() {
        // A degree-4 polynomial that we present as a length-33 Chebtech.
        int n = 33;
        double[] xs = ChebyshevPoints.secondKind(n);
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xs[i];
            v[i] = 1 + 2 * x + 3 * (2 * x * x - 1); // constants + T_1 + T_2
        }
        Chebtech t = Chebtech.fromValues(v);
        Chebtech s = t.simplify(1e-13);
        assertTrue(s.length() <= 4, "simplified length should be small, got " + s.length());
        assertClose(t.eval(0.42), s.eval(0.42), 1e-12, "eval preserved");
    }

    @Test
    public void testInvalidLengthThrows() {
        // Length 6 (n - 1 = 5) is not a power of two — should throw when
        // going through the FFT-based transform.
        double[] v = new double[6];
        assertThrows(IllegalArgumentException.class, () -> ChebTransform.vals2coeffs(v));
    }
}
