package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Rootfinding on the sign-change bracketing path (skips even-multiplicity
 * roots, per the documented MVP caveat); global min/max via critical
 * points + endpoints; and L^1 / L^2 / L^inf norms against analytic
 * references.
 */
public class ChebfunRootsExtremaNormsTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testRootsOfSinOnZeroToPi() {
        // sin has roots at 0 and pi in [0, pi]. Endpoints — bracketing will
        // catch them because sin(0) = 0 and sin(pi) = 0 to rounding.
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, Math.PI));
        double[] r = f.roots();
        // We expect roots near 0 and pi. Endpoint-detection is inexact due
        // to floating rounding — 0 may not be exactly zero after Chebyshev
        // fitting, but should be within 1e-13.
        boolean has0 = false, hasPi = false;
        for (double x : r) {
            if (Math.abs(x - 0.0) < 1e-10) has0 = true;
            if (Math.abs(x - Math.PI) < 1e-10) hasPi = true;
        }
        assertTrue(has0 || hasPi, "expected sin roots at 0 or pi, got " + java.util.Arrays.toString(r));
    }

    @Test
    public void testRootsOfSinOnInterior() {
        // sin has roots at pi and 2 pi inside [pi/2, 5 pi / 2].
        Chebfun f = new Chebfun(Math::sin, new Domain(Math.PI / 2, 5 * Math.PI / 2));
        double[] r = f.roots();
        assertEquals(2, r.length, "expected 2 sin roots, got " + java.util.Arrays.toString(r));
        assertClose(Math.PI, r[0], 1e-12, "root pi");
        assertClose(2 * Math.PI, r[1], 1e-12, "root 2 pi");
    }

    @Test
    public void testRootsOfPolynomial() {
        // f(x) = (x - 0.3)(x + 0.7) on [-1, 1]. Roots at 0.3 and -0.7.
        Chebfun f = new Chebfun(x -> (x - 0.3) * (x + 0.7), new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertEquals(2, r.length, "expected 2 roots, got " + java.util.Arrays.toString(r));
        assertClose(-0.7, r[0], 1e-12, "root -0.7");
        assertClose(0.3, r[1], 1e-12, "root 0.3");
    }

    @Test
    public void testMinMaxOfSin() {
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, 2.0 * Math.PI));
        Chebfun.Extremum lo = f.min();
        Chebfun.Extremum hi = f.max();
        assertClose(-1.0, lo.value(), 1e-12, "min value");
        assertClose(3 * Math.PI / 2, lo.location(), 1e-8, "min location");
        assertClose(1.0, hi.value(), 1e-12, "max value");
        assertClose(Math.PI / 2, hi.location(), 1e-8, "max location");
    }

    @Test
    public void testMinMaxOfPolynomial() {
        // f(x) = x^2 on [-2, 3]. Min = 0 at x = 0, max = 9 at x = 3.
        Chebfun f = new Chebfun(x -> x * x, new Domain(-2.0, 3.0));
        assertClose(0.0, f.min().value(), 1e-12, "min x^2");
        assertClose(0.0, f.min().location(), 1e-8, "argmin x^2");
        assertClose(9.0, f.max().value(), 1e-12, "max x^2");
        assertClose(3.0, f.max().location(), 1e-12, "argmax x^2");
    }

    @Test
    public void testNormInf() {
        // norm_inf(cos on [0, 2 pi]) = 1.
        Chebfun f = new Chebfun(Math::cos, new Domain(0.0, 2 * Math.PI));
        assertClose(1.0, f.normInf(), 1e-12, "|| cos ||_inf");
    }

    @Test
    public void testNorm2OfSinIsSqrtPi() {
        // ||sin||_2^2 on [0, 2 pi] = integral_0^{2 pi} sin^2 = pi.
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, 2 * Math.PI));
        assertClose(Math.sqrt(Math.PI), f.norm2(), 1e-12, "|| sin ||_2");
    }

    @Test
    public void testNorm1OfSinOnZeroToTwoPi() {
        // integral_0^{2 pi} |sin| = 4.
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, 2 * Math.PI));
        assertClose(4.0, f.norm1(), 1e-10, "|| sin ||_1");
    }

    @Test
    public void testNorm1OfPositiveFunction() {
        // ||exp||_1 on [0, 1] = e - 1 (no roots).
        Chebfun f = new Chebfun(Math::exp, new Domain(0.0, 1.0));
        assertClose(Math.E - 1, f.norm1(), 1e-12, "|| exp ||_1");
    }
}
