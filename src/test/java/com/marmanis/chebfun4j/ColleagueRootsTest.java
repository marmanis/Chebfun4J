package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests exercising the capabilities the colleague-matrix rootfinder
 * adds over the old sign-change bracketing: even-multiplicity roots
 * (x^2, (x-r)^2), interior double zeros, high-degree Chebyshev-polynomial
 * roots (which trigger the subdivision path once length exceeds the
 * colleague threshold), and known transcendental-function root sets.
 */
public class ColleagueRootsTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testEvenMultiplicityRootAtZero() {
        // x^2 has a double root at 0. Old bracketer missed this; colleague
        // matrix reports it as a single distinct root.
        Chebfun f = new Chebfun(x -> x * x, new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertEquals(1, r.length, "x^2 should give one distinct root, got " + java.util.Arrays.toString(r));
        assertClose(0.0, r[0], 1e-8, "x^2 root");
    }

    @Test
    public void testEvenMultiplicityShiftedRoot() {
        // (x - 0.3)^2 on [-1, 1]: double root at 0.3.
        Chebfun f = new Chebfun(x -> (x - 0.3) * (x - 0.3), new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertEquals(1, r.length, "double root: " + java.util.Arrays.toString(r));
        assertClose(0.3, r[0], 1e-6, "root 0.3");
    }

    @Test
    public void testChebyshevPolynomialT5() {
        // T_5(x) = 16 x^5 - 20 x^3 + 5 x has roots at cos((2k-1) pi / 10),
        // k=1..5. All simple roots inside [-1, 1].
        Chebfun f = new Chebfun(x -> 16*x*x*x*x*x - 20*x*x*x + 5*x, new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertEquals(5, r.length, "T_5 has 5 roots, got " + java.util.Arrays.toString(r));
        double[] want = new double[5];
        for (int k = 0; k < 5; k++) want[k] = Math.cos((2 * (k + 1) - 1) * Math.PI / 10.0);
        java.util.Arrays.sort(want);
        for (int k = 0; k < 5; k++) assertClose(want[k], r[k], 1e-12, "T_5 root " + k);
    }

    @Test
    public void testHighDegreePolynomialTriggersSubdivision() {
        // sin(10 pi x) on [-1, 1] has 21 roots at x = k/10 for k = -10..10.
        // Its Chebyshev expansion has ~50-70 terms, near the split threshold.
        Chebfun f = new Chebfun(x -> Math.sin(10 * Math.PI * x), new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertEquals(21, r.length, "sin(10 pi x) has 21 roots, got " + r.length);
        for (int k = 0; k < 21; k++) {
            double want = (k - 10) / 10.0;
            assertClose(want, r[k], 1e-8, "sin(10 pi x) root " + k);
        }
    }

    @Test
    public void testVeryHighDegreeForcesSubdivision() {
        // A rapidly oscillating function that needs enough Chebyshev
        // coefficients to definitely trigger the subdivision path.
        Chebfun f = new Chebfun(x -> Math.sin(30 * x), new Domain(-1.0, 1.0));
        double[] r = f.roots();
        // sin(30 x) has roots at x = k pi / 30 for integer k with |k pi / 30| <= 1
        // -> k = -9..9, that's 19 roots (0, and 18 non-zero).
        assertEquals(19, r.length, "sin(30 x) has 19 roots, got " + r.length);
        for (int k = 0; k < 19; k++) {
            double want = (k - 9) * Math.PI / 30.0;
            // Colleague eigenvalues are accurate to ~sqrt(eps) near densely-
            // packed roots; 1e-6 is a fair MVP tolerance.
            assertClose(want, r[k], 1e-6, "sin(30 x) root " + k);
        }
    }

    @Test
    public void testMinOfXSquaredWorks() {
        // Regression: with the old rootfinder, min(x^2) still worked because
        // 2x had a simple root at 0, but let's re-verify it still does.
        Chebfun f = new Chebfun(x -> x * x, new Domain(-2.0, 3.0));
        Chebfun.Extremum lo = f.min();
        assertClose(0.0, lo.value(), 1e-12, "min value");
        assertClose(0.0, lo.location(), 1e-8, "min location");
    }

    @Test
    public void testNoRootsFunctionReturnsEmpty() {
        // exp(x) has no zeros.
        Chebfun f = new Chebfun(Math::exp, new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertEquals(0, r.length, "exp has no roots, got " + java.util.Arrays.toString(r));
    }

    @Test
    public void testCubicWithTripleRootAtOrigin() {
        // x^3 has a triple root at 0. Colleague accuracy at a triple root
        // degrades to eps^{1/3} ~= 6e-6, per the standard sensitivity
        // theorem for repeated eigenvalues — so we accept up to 1e-4.
        Chebfun f = new Chebfun(x -> x * x * x, new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertTrue(r.length >= 1, "expected at least one root");
        for (double x : r) assertClose(0.0, x, 1e-4, "cubic root");
    }
}
