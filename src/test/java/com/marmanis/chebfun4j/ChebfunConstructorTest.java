package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Adaptive construction: resolves smooth functions to close to machine
 * precision, uses fewer coefficients for lower-degree functions (simplify),
 * and gives correct pointwise values inside the domain.
 */
public class ChebfunConstructorTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testResolvesExpToMachinePrecision() {
        Chebfun f = new Chebfun(Math::exp, new Domain(-1.0, 1.0));
        assertTrue(f.length() < 20, "exp on [-1,1] should resolve in < 20 coefficients, got " + f.length());
        double[] test = {-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0};
        for (double x : test) {
            assertClose(Math.exp(x), f.feval(x), 1e-13, "exp @ " + x);
        }
    }

    @Test
    public void testResolvesSinCosOnLargerDomain() {
        Domain d = new Domain(0.0, 2.0 * Math.PI);
        Chebfun f = new Chebfun(Math::sin, d);
        double[] test = {0.0, Math.PI / 6, Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI};
        for (double x : test) {
            assertClose(Math.sin(x), f.feval(x), 1e-12, "sin @ " + x);
        }
    }

    @Test
    public void testConstantResolvesToLengthOne() {
        Chebfun f = new Chebfun(x -> 3.7, new Domain(-2.0, 5.0));
        assertEquals(1, f.length(), "constant should simplify to length 1");
        assertClose(3.7, f.feval(-2.0), 1e-15, "const endpoint");
        assertClose(3.7, f.feval(1.234), 1e-15, "const interior");
    }

    @Test
    public void testLinearResolvesToLengthTwo() {
        // f(x) = 2x + 3 on [0, 10].
        Chebfun f = new Chebfun(x -> 2 * x + 3, new Domain(0.0, 10.0));
        assertEquals(2, f.length(), "linear should simplify to length 2");
        assertClose(3.0, f.feval(0.0), 1e-13, "f(0)");
        assertClose(23.0, f.feval(10.0), 1e-13, "f(10)");
        assertClose(13.0, f.feval(5.0), 1e-13, "f(5)");
    }

    @Test
    public void testQuadraticResolvesToLengthThree() {
        // f(x) = x^2 on [-3, 3].
        Chebfun f = new Chebfun(x -> x * x, new Domain(-3.0, 3.0));
        assertEquals(3, f.length(), "quadratic should simplify to length 3");
        assertClose(4.0, f.feval(2.0), 1e-12, "f(2)");
        assertClose(0.0, f.feval(0.0), 1e-12, "f(0)");
    }

    @Test
    public void testResolvesRungeOnMinusOneToOne() {
        // Runge's function 1/(1 + 25 x^2) is smooth and analytic on [-1,1],
        // so chebfun handles it fine (Runge phenomenon only bites equispaced
        // interpolation, not Chebyshev).
        Chebfun f = new Chebfun(x -> 1.0 / (1 + 25 * x * x), new Domain(-1.0, 1.0));
        double[] test = {-0.9, -0.5, 0.0, 0.3, 0.7};
        for (double x : test) {
            double want = 1.0 / (1 + 25 * x * x);
            assertClose(want, f.feval(x), 1e-12, "runge @ " + x);
        }
    }

    @Test
    public void testDomainMapping() {
        // On [2, 6], f(x) = x should still resolve to length 2 and give
        // correct values.
        Chebfun f = new Chebfun(x -> x, new Domain(2.0, 6.0));
        assertEquals(2, f.length());
        assertClose(2.0, f.feval(2.0), 1e-13, "f(a)");
        assertClose(6.0, f.feval(6.0), 1e-13, "f(b)");
        assertClose(4.0, f.feval(4.0), 1e-13, "f(mid)");
    }
}
