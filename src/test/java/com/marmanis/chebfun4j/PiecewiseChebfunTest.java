package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The splitting-on adaptive constructor and every operation that now has to
 * work across pieces: evaluation with a breakpoint, sum/diff over multiple
 * pieces, arithmetic that merges breakpoints, and roots that need to
 * dedup across piece boundaries.
 */
public class PiecewiseChebfunTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testAbsSplitsIntoTwoPieces() {
        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
        assertTrue(f.numPieces() >= 2, "|x| should split, got " + f.numPieces() + " pieces");
        for (double x : new double[]{-0.9, -0.5, 0.0, 0.4, 0.9}) {
            assertClose(Math.abs(x), f.feval(x), 1e-12, "|x| @ " + x);
        }
        // A breakpoint should land near 0.
        double[] bp = f.breakpoints();
        boolean nearZero = false;
        for (int i = 1; i < bp.length - 1; i++) {
            if (Math.abs(bp[i]) < 1e-6) nearZero = true;
        }
        assertTrue(nearZero, "expected a breakpoint near 0, got " + java.util.Arrays.toString(bp));
    }

    @Test
    public void testSignFunctionSplits() {
        // sign(x) has a jump at 0.
        Chebfun f = new Chebfun(x -> Math.signum(x), new Domain(-1.0, 1.0));
        assertTrue(f.numPieces() >= 2, "sign should split, got " + f.numPieces());
        assertClose(-1.0, f.feval(-0.5), 1e-10, "sign(-)");
        assertClose(+1.0, f.feval(+0.5), 1e-10, "sign(+)");
    }

    @Test
    public void testSumOfAbsMatchesAnalytic() {
        // integral_{-1}^{1} |x| dx = 1.
        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
        assertClose(1.0, f.sum(), 1e-11, "int |x|");
    }

    @Test
    public void testDiffOfAbsIsSign() {
        // Away from the kink, d|x|/dx = sign(x).
        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
        Chebfun df = f.diff();
        assertClose(-1.0, df.feval(-0.5), 1e-10, "d|x|/dx at -0.5");
        assertClose(+1.0, df.feval(+0.5), 1e-10, "d|x|/dx at +0.5");
    }

    @Test
    public void testMinMaxOfAbs() {
        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
        Chebfun.Extremum lo = f.min();
        Chebfun.Extremum hi = f.max();
        assertClose(0.0, lo.value(), 1e-11, "min |x|");
        assertClose(1.0, hi.value(), 1e-11, "max |x|");
    }

    @Test
    public void testL1OfAbsIsOne() {
        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
        assertClose(1.0, f.norm1(), 1e-11, "|| |x| ||_1");
    }

    @Test
    public void testArithmeticMergesBreakpoints() {
        // f = |x - 0.3| has a breakpoint near 0.3.
        // g = |x + 0.5| has a breakpoint near -0.5.
        // f + g should have both breakpoints.
        Chebfun f = new Chebfun(x -> Math.abs(x - 0.3), new Domain(-1.0, 1.0));
        Chebfun g = new Chebfun(x -> Math.abs(x + 0.5), new Domain(-1.0, 1.0));
        Chebfun sum = f.plus(g);
        assertTrue(sum.numPieces() >= 2,
            "sum should have at least 2 pieces, got " + sum.numPieces());
        for (double x : new double[]{-0.9, -0.2, 0.1, 0.8}) {
            double want = Math.abs(x - 0.3) + Math.abs(x + 0.5);
            assertClose(want, sum.feval(x), 1e-11, "sum @ " + x);
        }
    }

    @Test
    public void testSmoothCaseStaysOnePiece() {
        // Regression: a smooth function should still resolve as a single
        // piece and pass all the old operations.
        Chebfun f = new Chebfun(Math::exp, new Domain(-1.0, 1.0));
        assertEquals(1, f.numPieces(), "smooth exp is one piece");
        assertClose(Math.E - 1 / Math.E, f.sum(), 1e-13, "int exp");
    }

    @Test
    public void testRootsAcrossPieces() {
        // sin(pi x) has roots at -1, 0, 1 on [-1, 1]. Smooth, one piece.
        Chebfun f = new Chebfun(x -> Math.sin(Math.PI * x), new Domain(-1.0, 1.0));
        double[] r = f.roots();
        assertTrue(r.length >= 3, "expected 3 roots got " + r.length);
        // Verify 0 is present.
        boolean hasZero = false;
        for (double x : r) if (Math.abs(x) < 1e-10) hasZero = true;
        assertTrue(hasZero, "expected root at 0");
    }

    @Test
    public void testTwoPieceCumsumIsContinuous() {
        // cumsum of |x| = x^2/2 * sign(x) + constant. Values on both sides
        // of 0 must agree at 0 (cumsum is continuous).
        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));
        Chebfun F = f.cumsum();
        assertClose(0.0, F.feval(-1.0), 1e-12, "F(a) = 0");
        double leftAtMid = F.feval(0.0 - 1e-9);
        double rightAtMid = F.feval(0.0 + 1e-9);
        assertClose(leftAtMid, rightAtMid, 1e-8, "F continuous at 0");
        // Also F(1) should equal sum(f) = 1.
        assertClose(1.0, F.feval(1.0), 1e-11, "F(b) = sum(f)");
    }
}
