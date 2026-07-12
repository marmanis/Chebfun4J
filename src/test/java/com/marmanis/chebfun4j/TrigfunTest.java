package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Trigfun exercises the periodic path: Fourier construction, evaluation,
 * exact Fourier-identity products (sin*cos, sin^2), differentiation as
 * frequency-domain scaling, definite integrals over one period, and
 * simplification of a low-frequency signal.
 */
public class TrigfunTest {

    private static final Domain TWO_PI = new Domain(0.0, 2 * Math.PI);

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSinResolvesToSmallLength() {
        Trigfun f = new Trigfun(Math::sin, TWO_PI);
        // Pure sin needs Fourier length at most ~16.
        assertTrue(f.length() <= 32, "sin should resolve short, got " + f.length());
        for (double x : new double[]{0.0, Math.PI / 4, Math.PI / 2, Math.PI, 3 * Math.PI / 2}) {
            assertClose(Math.sin(x), f.feval(x), 1e-13, "sin @ " + x);
        }
    }

    @Test
    public void testCosResolves() {
        Trigfun f = new Trigfun(Math::cos, TWO_PI);
        for (double x : new double[]{0.0, Math.PI / 3, Math.PI, 5 * Math.PI / 4}) {
            assertClose(Math.cos(x), f.feval(x), 1e-13, "cos @ " + x);
        }
    }

    @Test
    public void testSinTimesCosIdentity() {
        // sin(x) cos(x) = sin(2 x) / 2.
        Trigfun s = new Trigfun(Math::sin, TWO_PI);
        Trigfun c = new Trigfun(Math::cos, TWO_PI);
        Trigfun prod = s.times(c);
        for (double x : new double[]{0.1, 0.7, 1.3, 2.0, 3.5}) {
            assertClose(0.5 * Math.sin(2 * x), prod.feval(x), 1e-12, "sin*cos identity @ " + x);
        }
    }

    @Test
    public void testSinSquaredMeanIsHalf() {
        // integral_0^{2 pi} sin^2 = pi.
        Trigfun s = new Trigfun(Math::sin, TWO_PI);
        double integral = s.times(s).sum();
        assertClose(Math.PI, integral, 1e-12, "int sin^2");
    }

    @Test
    public void testDiffOfSinIsCos() {
        Trigfun f = new Trigfun(Math::sin, TWO_PI);
        Trigfun df = f.diff();
        for (double x : new double[]{0.0, Math.PI / 3, Math.PI, 7 * Math.PI / 4}) {
            assertClose(Math.cos(x), df.feval(x), 1e-12, "d/dx sin @ " + x);
        }
    }

    @Test
    public void testIntegralOverPeriod() {
        // integral of any zero-mean periodic function over the period = 0.
        Trigfun f = new Trigfun(Math::sin, TWO_PI);
        assertClose(0.0, f.sum(), 1e-13, "int sin over full period");
    }

    @Test
    public void testConstantHasSingleCoefficient() {
        Trigfun f = new Trigfun(x -> 2.5, TWO_PI);
        // A constant simplifies down to length 2 (minimum representable length).
        assertTrue(f.length() <= 4, "constant should simplify small, got " + f.length());
        assertClose(2.5, f.feval(1.2345), 1e-13, "const eval");
        // integral over period = 2 pi * 2.5 = 5 pi.
        assertClose(5 * Math.PI, f.sum(), 1e-12, "int const");
    }

    @Test
    public void testDomainMapping() {
        // sin(pi (x - 3) / 5) on [3, 13] is periodic with period 10.
        Domain d = new Domain(3.0, 13.0);
        Trigfun f = new Trigfun(x -> Math.sin(Math.PI * (x - 3) / 5.0), d);
        for (double x : new double[]{4.0, 5.5, 8.0, 11.5}) {
            double want = Math.sin(Math.PI * (x - 3) / 5.0);
            assertClose(want, f.feval(x), 1e-12, "mapped sin @ " + x);
        }
    }
}
