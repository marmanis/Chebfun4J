package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Chebfun2: ACA constructor rank, evaluation, integration (double + marginal),
 * partial derivatives, arithmetic.
 */
public class Chebfun2Test {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    private static final Rectangle UNIT = Rectangle.unit();

    @Test
    public void testSeparableFunctionIsRankOne() {
        // sin(x) cos(y) is rank 1 by construction.
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), UNIT);
        assertEquals(1, f.rank(), "sin(x)cos(y) should be rank 1");
        for (double x : new double[]{-0.7, 0.0, 0.5}) {
            for (double y : new double[]{-0.4, 0.0, 0.8}) {
                assertClose(Math.sin(x) * Math.cos(y), f.feval(x, y), 1e-12, "@ (" + x + "," + y + ")");
            }
        }
    }

    @Test
    public void testExpXPlusYIsRankOne() {
        // e^{x+y} = e^x e^y is rank 1.
        Chebfun2 f = new Chebfun2((x, y) -> Math.exp(x + y), UNIT);
        assertEquals(1, f.rank(), "e^{x+y} should be rank 1");
        for (double x : new double[]{-0.5, 0.2, 0.9}) {
            for (double y : new double[]{-0.3, 0.4, 0.7}) {
                assertClose(Math.exp(x + y), f.feval(x, y), 1e-11, "@ (" + x + "," + y + ")");
            }
        }
    }

    @Test
    public void testDoubleIntegralOfSinXCosY() {
        // integral_{-1}^{1} integral_{-1}^{1} sin(x) cos(y) dx dy
        //  = (integral sin x) * (integral cos y) = 0 * 2 sin(1) = 0.
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), UNIT);
        assertClose(0.0, f.sum2(), 1e-13, "sum2 sin cos");
    }

    @Test
    public void testDoubleIntegralOfConstantMatchesArea() {
        Rectangle r = new Rectangle(0.0, 2.0, -1.0, 3.0); // area = 2 * 4 = 8.
        Chebfun2 f = new Chebfun2((x, y) -> 3.5, r);
        assertClose(3.5 * 8.0, f.sum2(), 1e-11, "constant sum2");
    }

    @Test
    public void testMarginalIntegralOverX() {
        // integral_{-1}^{1} sin(x) cos(y) dx = 0 * cos(y) = 0 for all y.
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), UNIT);
        Chebfun marginal = f.sum(0);
        for (double y : new double[]{-0.5, 0.0, 0.7}) {
            assertClose(0.0, marginal.feval(y), 1e-12, "marginal(y) @ " + y);
        }
    }

    @Test
    public void testMarginalIntegralNonZero() {
        // integral_{-1}^{1} e^x sin(y) dx = sin(y) * (e - 1/e).
        Chebfun2 f = new Chebfun2((x, y) -> Math.exp(x) * Math.sin(y), UNIT);
        Chebfun marginal = f.sum(0);
        double scale = Math.E - 1 / Math.E;
        for (double y : new double[]{-0.9, -0.3, 0.4, 0.8}) {
            assertClose(scale * Math.sin(y), marginal.feval(y), 1e-11, "marginal @ y=" + y);
        }
    }

    @Test
    public void testPartialX() {
        // d/dx (x^2 + y^2) = 2x, independent of y.
        Chebfun2 f = new Chebfun2((x, y) -> x * x + y * y, UNIT);
        Chebfun2 fx = f.partialX();
        for (double x : new double[]{-0.7, 0.0, 0.5}) {
            for (double y : new double[]{-0.4, 0.0, 0.8}) {
                assertClose(2 * x, fx.feval(x, y), 1e-11, "partialX @ (" + x + "," + y + ")");
            }
        }
    }

    @Test
    public void testPartialY() {
        // d/dy (sin x + cos y) = -sin y.
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) + Math.cos(y), UNIT);
        Chebfun2 fy = f.partialY();
        for (double x : new double[]{-0.5, 0.2}) {
            for (double y : new double[]{-0.6, 0.1, 0.8}) {
                assertClose(-Math.sin(y), fy.feval(x, y), 1e-11, "partialY @ (" + x + "," + y + ")");
            }
        }
    }

    @Test
    public void testArithmetic() {
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), UNIT);
        Chebfun2 g = new Chebfun2((x, y) -> Math.cos(x) * Math.sin(y), UNIT);
        Chebfun2 sum = f.plus(g);
        Chebfun2 diff = f.minus(g);
        for (double x : new double[]{-0.5, 0.0, 0.6}) {
            for (double y : new double[]{-0.3, 0.0, 0.5}) {
                double wantSum  = Math.sin(x) * Math.cos(y) + Math.cos(x) * Math.sin(y);
                double wantDiff = Math.sin(x) * Math.cos(y) - Math.cos(x) * Math.sin(y);
                assertClose(wantSum,  sum.feval(x, y),  1e-11, "sum @ ("+x+","+y+")");
                assertClose(wantDiff, diff.feval(x, y), 1e-11, "diff @ ("+x+","+y+")");
            }
        }
    }

    @Test
    public void testScalarTimes() {
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), UNIT);
        Chebfun2 g = f.times(3.7);
        for (double x : new double[]{-0.5, 0.4}) {
            for (double y : new double[]{-0.3, 0.6}) {
                assertClose(3.7 * Math.sin(x) * Math.cos(y), g.feval(x, y), 1e-11, "scalar times");
            }
        }
    }

    @Test
    public void testMappedRectangle() {
        // On [1, 3] x [2, 4], f(x, y) = x + y (rank 2 separable).
        Rectangle r = new Rectangle(1.0, 3.0, 2.0, 4.0);
        Chebfun2 f = new Chebfun2((x, y) -> x + y, r);
        assertTrue(f.rank() <= 2, "x + y rank should be at most 2, got " + f.rank());
        for (double x : new double[]{1.5, 2.0, 2.5}) {
            for (double y : new double[]{2.5, 3.0, 3.5}) {
                assertClose(x + y, f.feval(x, y), 1e-11, "@ ("+x+","+y+")");
            }
        }
    }

    @Test
    public void testChebfun2Product() {
        // sin(x)cos(y) * exp(x+y) = e^x sin(x) e^y cos(y). Rank 1.
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), UNIT);
        Chebfun2 g = new Chebfun2((x, y) -> Math.exp(x + y), UNIT);
        Chebfun2 prod = f.times(g);
        for (double x : new double[]{-0.7, 0.0, 0.5}) {
            for (double y : new double[]{-0.4, 0.0, 0.8}) {
                double want = Math.sin(x) * Math.cos(y) * Math.exp(x + y);
                assertClose(want, prod.feval(x, y), 1e-11, "prod @ (" + x + "," + y + ")");
            }
        }
    }

    @Test
    public void testChebfun2ProductRankOneTimesRankOne() {
        // e^x * e^y * (e^x * e^y) = e^{2x} * e^{2y}. Still rank 1.
        Chebfun2 f = new Chebfun2((x, y) -> Math.exp(x) * Math.exp(y), UNIT);
        Chebfun2 sq = f.times(f);
        assertTrue(sq.rank() <= 2, "e^{2x} e^{2y} should stay low rank, got " + sq.rank());
        for (double x : new double[]{-0.5, 0.3}) {
            for (double y : new double[]{-0.2, 0.6}) {
                assertClose(Math.exp(2 * x) * Math.exp(2 * y), sq.feval(x, y), 1e-11, "prod @ ("+x+","+y+")");
            }
        }
    }

    @Test
    public void testInvalidRectangleThrows() {
        try {
            new Rectangle(1.0, 1.0, 0.0, 1.0);
            throw new AssertionError("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {}
    }
}
