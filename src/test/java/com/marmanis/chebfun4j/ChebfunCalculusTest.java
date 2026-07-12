package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

/**
 * Calculus: sum matches known analytic integrals; diff matches known
 * derivatives; cumsum(f)(b) equals sum(f) and cumsum(f)(a) equals 0;
 * diff(cumsum(f)) == f.
 */
public class ChebfunCalculusTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSumOfSinIsZeroOnFullPeriod() {
        // integral_0^{2 pi} sin = 0.
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, 2.0 * Math.PI));
        assertClose(0.0, f.sum(), 1e-13, "int sin over full period");
    }

    @Test
    public void testSumOfCosIsAnalytic() {
        // integral_0^{pi/2} cos = 1.
        Chebfun f = new Chebfun(Math::cos, new Domain(0.0, Math.PI / 2));
        assertClose(1.0, f.sum(), 1e-13, "int cos");
    }

    @Test
    public void testSumOfExp() {
        // integral_0^1 e^x = e - 1.
        Chebfun f = new Chebfun(Math::exp, new Domain(0.0, 1.0));
        assertClose(Math.E - 1, f.sum(), 1e-13, "int exp");
    }

    @Test
    public void testSumOfPolynomial() {
        // integral_{-2}^{3} (x^3 - x) dx = [x^4/4 - x^2/2]_{-2}^{3}
        //   = (81/4 - 9/2) - (16/4 - 4/2) = 63/4 - 2 = 55/4 = 13.75.
        Chebfun f = new Chebfun(x -> x * x * x - x, new Domain(-2.0, 3.0));
        assertClose(13.75, f.sum(), 1e-12, "int poly");
    }

    @Test
    public void testDiffOfSinIsCos() {
        Domain d = new Domain(-1.0, 1.0);
        Chebfun f = new Chebfun(Math::sin, d);
        Chebfun df = f.diff();
        for (double x : new double[]{-0.9, -0.3, 0.0, 0.3, 0.9}) {
            assertClose(Math.cos(x), df.feval(x), 1e-12, "d/dx sin @ " + x);
        }
    }

    @Test
    public void testDiffOfExpIsExp() {
        Chebfun f = new Chebfun(Math::exp, new Domain(-1.0, 1.0));
        Chebfun df = f.diff();
        for (double x : new double[]{-0.7, 0.0, 0.5, 0.9}) {
            assertClose(Math.exp(x), df.feval(x), 1e-12, "d/dx exp @ " + x);
        }
    }

    @Test
    public void testDiffOfPolynomial() {
        // d/dx (x^3 - 2 x + 1) = 3 x^2 - 2.
        Chebfun f = new Chebfun(x -> x * x * x - 2 * x + 1, new Domain(-2.0, 2.0));
        Chebfun df = f.diff();
        for (double x : new double[]{-1.5, -0.5, 0.5, 1.5}) {
            assertClose(3 * x * x - 2, df.feval(x), 1e-12, "d poly @ " + x);
        }
    }

    @Test
    public void testCumsumThenDiffIsIdentity() {
        Chebfun f = new Chebfun(Math::cos, new Domain(0.0, 2.0));
        Chebfun back = f.cumsum().diff();
        for (double x : new double[]{0.1, 0.5, 1.0, 1.5, 1.9}) {
            assertClose(Math.cos(x), back.feval(x), 1e-11, "diff(cumsum) @ " + x);
        }
    }

    @Test
    public void testCumsumBoundaryConditions() {
        // F(a) = 0, F(b) = sum(f).
        Domain d = new Domain(0.0, Math.PI);
        Chebfun f = new Chebfun(Math::sin, d);
        Chebfun F = f.cumsum();
        assertClose(0.0, F.feval(0.0), 1e-12, "F(a) = 0");
        assertClose(f.sum(), F.feval(Math.PI), 1e-12, "F(b) = sum(f)");
        // integral of sin from 0 to pi = 2.
        assertClose(2.0, F.feval(Math.PI), 1e-12, "int_0^pi sin");
    }
}
