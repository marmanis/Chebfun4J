package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Arithmetic: sums, products, scalar operations produce the mathematically
 * correct chebfun (verified pointwise against the analytic combined
 * function). Also mismatched domains throw.
 */
public class ChebfunArithmeticTest {

    private static final Domain UNIT = new Domain(-1.0, 1.0);

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testPlusAndMinusMatchAnalytic() {
        Chebfun f = new Chebfun(Math::sin, UNIT);
        Chebfun g = new Chebfun(Math::cos, UNIT);
        Chebfun sum = f.plus(g);
        Chebfun diff = f.minus(g);
        double[] test = {-0.9, -0.3, 0.0, 0.3, 0.9};
        for (double x : test) {
            assertClose(Math.sin(x) + Math.cos(x), sum.feval(x), 1e-13, "sum @ " + x);
            assertClose(Math.sin(x) - Math.cos(x), diff.feval(x), 1e-13, "diff @ " + x);
        }
    }

    @Test
    public void testTimesMatchAnalytic() {
        // f(x) = e^x, g(x) = sin(x); product should equal e^x sin(x).
        Chebfun f = new Chebfun(Math::exp, UNIT);
        Chebfun g = new Chebfun(Math::sin, UNIT);
        Chebfun prod = f.times(g);
        double[] test = {-0.9, -0.3, 0.0, 0.3, 0.7, 0.9};
        for (double x : test) {
            assertClose(Math.exp(x) * Math.sin(x), prod.feval(x), 1e-12, "prod @ " + x);
        }
    }

    @Test
    public void testTimesTwoPolynomialsExact() {
        // (x^2 - 1) * (x + 2). Result should have length exactly 4 (degree 3).
        Chebfun f = new Chebfun(x -> x * x - 1, UNIT);
        Chebfun g = new Chebfun(x -> x + 2, UNIT);
        Chebfun prod = f.times(g);
        assertEquals(4, prod.length(), "cubic product should simplify to length 4");
        double[] test = {-0.9, -0.5, 0.0, 0.5, 0.9};
        for (double x : test) {
            double want = (x * x - 1) * (x + 2);
            assertClose(want, prod.feval(x), 1e-13, "poly prod @ " + x);
        }
    }

    @Test
    public void testScalarPlusAndTimes() {
        Chebfun f = new Chebfun(Math::sin, UNIT);
        Chebfun shifted = f.plus(3.0);
        Chebfun scaled = f.times(2.5);
        double[] test = {-0.7, 0.0, 0.4};
        for (double x : test) {
            assertClose(Math.sin(x) + 3.0, shifted.feval(x), 1e-13, "shift @ " + x);
            assertClose(2.5 * Math.sin(x), scaled.feval(x), 1e-13, "scale @ " + x);
        }
    }

    @Test
    public void testNegate() {
        Chebfun f = new Chebfun(Math::sin, UNIT);
        Chebfun neg = f.negate();
        for (double x : new double[]{-0.8, -0.1, 0.5}) {
            assertClose(-Math.sin(x), neg.feval(x), 1e-13, "neg @ " + x);
        }
    }

    @Test
    public void testDomainMismatchThrows() {
        Chebfun f = new Chebfun(Math::sin, new Domain(-1.0, 1.0));
        Chebfun g = new Chebfun(Math::cos, new Domain(0.0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> f.plus(g));
        assertThrows(IllegalArgumentException.class, () -> f.times(g));
    }

    @Test
    public void testArithmeticOnMappedDomain() {
        // Product on [2, 5]: f(x) = x, g(x) = x - 3.
        Domain d = new Domain(2.0, 5.0);
        Chebfun f = new Chebfun(x -> x, d);
        Chebfun g = new Chebfun(x -> x - 3, d);
        Chebfun prod = f.times(g);
        for (double x : new double[]{2.5, 3.0, 4.0, 4.9}) {
            assertClose(x * (x - 3), prod.feval(x), 1e-12, "mapped prod @ " + x);
        }
    }
}
