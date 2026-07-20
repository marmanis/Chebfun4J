package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

/**
 * Iteration 6 API completions: {@code compose}, {@code divide}, {@code pow},
 * {@code abs}, {@code sum(a, b)}. Each test picks a case with a known
 * closed-form answer and checks agreement to spectral precision.
 */
public class ChebfunApiCompletionsTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    // -----------------------------------------------------------------
    // sum(a, b)
    // -----------------------------------------------------------------

    @Test
    public void testSumOverSubintervalMatchesClosedForm() {
        // integral_0^{π/2} sin(x) dx = 1.
        Chebfun s = new Chebfun(Math::sin, new Domain(0.0, Math.PI));
        assertClose(1.0, s.sum(0.0, Math.PI / 2), 1e-12, "int_0^(π/2) sin");
        // integral_{π/2}^π sin(x) dx = 1 as well.
        assertClose(1.0, s.sum(Math.PI / 2, Math.PI), 1e-12, "int_(π/2)^π sin");
    }

    @Test
    public void testSumReversedEndpointsFlipsSign() {
        Chebfun s = new Chebfun(Math::sin, new Domain(0.0, Math.PI));
        double forward = s.sum(0.5, 2.0);
        double reverse = s.sum(2.0, 0.5);
        assertClose(-forward, reverse, 1e-14, "sum(b,a) = -sum(a,b)");
    }

    @Test
    public void testSumFullDomainMatchesSumNoArgs() {
        Chebfun f = new Chebfun(x -> Math.exp(x), new Domain(-1.0, 2.0));
        double whole = f.sum();
        double via = f.sum(-1.0, 2.0);
        assertClose(whole, via, 1e-12, "sum() vs sum(a,b)");
    }

    // -----------------------------------------------------------------
    // divide
    // -----------------------------------------------------------------

    @Test
    public void testDivideRecoversTanFromSinCos() {
        // sin/cos = tan on [0, π/3] (well away from cos's zero at π/2).
        Domain d = new Domain(0.0, Math.PI / 3);
        Chebfun s = new Chebfun(Math::sin, d);
        Chebfun c = new Chebfun(Math::cos, d);
        Chebfun t = s.divide(c);
        for (double x : new double[]{0.1, 0.3, 0.7, 1.0}) {
            assertClose(Math.tan(x), t.feval(x), 1e-11, "tan @ " + x);
        }
    }

    @Test
    public void testDivideRequiresMatchingDomain() {
        Chebfun a = new Chebfun(x -> 1.0, new Domain(0.0, 1.0));
        Chebfun b = new Chebfun(x -> 1.0, new Domain(0.0, 2.0));
        try {
            a.divide(b);
            throw new AssertionError("expected domain mismatch to throw");
        } catch (IllegalArgumentException expected) {
            // OK
        }
    }

    // -----------------------------------------------------------------
    // pow
    // -----------------------------------------------------------------

    @Test
    public void testPowIntZeroIsConstantOne() {
        Chebfun f = new Chebfun(Math::sin, new Domain(-1.0, 1.0));
        Chebfun ones = f.pow(0);
        assertClose(1.0, ones.feval(0.3), 1e-14, "sin^0(0.3)");
        assertClose(1.0, ones.feval(-0.7), 1e-14, "sin^0(-0.7)");
    }

    @Test
    public void testPowIntFastExponentiationMatchesRepeatedMultiply() {
        // f = e^x on [0, 1], f^5 = e^(5x).
        Chebfun f = new Chebfun(Math::exp, new Domain(0.0, 1.0));
        Chebfun via = f.pow(5);
        for (double x : new double[]{0.1, 0.4, 0.7, 0.95}) {
            assertClose(Math.exp(5 * x), via.feval(x), 1e-10, "exp(5x) @ " + x);
        }
    }

    @Test
    public void testPowIntNegative() {
        // f = 2 + sin(x), f^{-1} = 1 / (2 + sin(x)) — bounded away from 0.
        Chebfun f = new Chebfun(x -> 2.0 + Math.sin(x), new Domain(-1.0, 1.0));
        Chebfun inv = f.pow(-1);
        for (double x : new double[]{-0.5, 0.0, 0.5}) {
            assertClose(1.0 / (2.0 + Math.sin(x)), inv.feval(x), 1e-11, "1/(2+sin) @ " + x);
        }
    }

    @Test
    public void testPowRealFractional() {
        // sqrt(1 + x²) on [-1, 1] via pow(0.5).
        Chebfun f = new Chebfun(x -> 1.0 + x * x, new Domain(-1.0, 1.0));
        Chebfun root = f.pow(0.5);
        for (double x : new double[]{-0.7, -0.2, 0.3, 0.9}) {
            assertClose(Math.sqrt(1.0 + x * x), root.feval(x), 1e-11, "sqrt(1+x²) @ " + x);
        }
    }

    // -----------------------------------------------------------------
    // abs
    // -----------------------------------------------------------------

    @Test
    public void testAbsResolvesPiecewiseAtInteriorZero() {
        // |x| on [-1, 1] — the classic splitting-on test case.
        Chebfun f = new Chebfun(x -> x, new Domain(-1.0, 1.0));
        Chebfun absF = f.abs();
        for (double x : new double[]{-0.9, -0.4, 0.0, 0.4, 0.9}) {
            assertClose(Math.abs(x), absF.feval(x), 1e-10, "|x| @ " + x);
        }
        // Piecewise splitting should introduce a breakpoint near 0.
        if (absF.numPieces() < 2) {
            throw new AssertionError("expected |x| to be piecewise, got numPieces=" + absF.numPieces());
        }
    }

    @Test
    public void testAbsOfPositiveFunctionIsOnePiece() {
        // f > 0 everywhere -> |f| = f, still one smooth piece.
        Chebfun f = new Chebfun(x -> 2.0 + x * x, new Domain(-1.0, 1.0));
        Chebfun absF = f.abs();
        for (double x : new double[]{-0.5, 0.0, 0.5}) {
            assertClose(f.feval(x), absF.feval(x), 1e-13, "|f| @ " + x);
        }
    }

    // -----------------------------------------------------------------
    // compose
    // -----------------------------------------------------------------

    @Test
    public void testComposeSinOfXSquaredMatchesDirect() {
        // f(y) = sin(y) on [0, 1], g(x) = x² on [0, 1]. f∘g(x) = sin(x²).
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, 1.0));
        Chebfun g = new Chebfun(x -> x * x, new Domain(0.0, 1.0));
        Chebfun h = f.compose(g);
        for (double x : new double[]{0.0, 0.3, 0.6, 0.9, 1.0}) {
            assertClose(Math.sin(x * x), h.feval(x), 1e-11, "sin(x²) @ " + x);
        }
    }

    @Test
    public void testComposeResultLivesOnInnerDomain() {
        Chebfun f = new Chebfun(Math::exp, new Domain(0.0, 2.0));
        Chebfun g = new Chebfun(x -> 0.5 * x + 0.5, new Domain(-1.0, 3.0));
        Chebfun h = f.compose(g);
        if (!h.domain().equalsDomain(g.domain())) {
            throw new AssertionError("compose result should live on inner domain " + g.domain()
                + ", got " + h.domain());
        }
    }

    // -----------------------------------------------------------------
    // samples
    // -----------------------------------------------------------------

    @Test
    public void testSamplesCoverDomainEndpoints() {
        Chebfun f = new Chebfun(Math::sin, new Domain(0.0, Math.PI));
        Chebfun.Samples s = f.samples(11);
        assertClose(0.0, s.x()[0], 1e-14, "x[0] = a");
        assertClose(Math.PI, s.x()[10], 1e-14, "x[n-1] = b");
        assertClose(Math.sin(0.0), s.y()[0], 1e-14, "y[0]");
        assertClose(Math.sin(Math.PI), s.y()[10], 1e-14, "y[n-1]");
    }

    @Test
    public void testSamplesUniformSpacing() {
        Chebfun f = new Chebfun(x -> x, new Domain(-1.0, 1.0));
        Chebfun.Samples s = f.samples(5);
        // Uniform spacing: 0.5 apart on [-1, 1] with 5 points.
        double[] expectX = {-1.0, -0.5, 0.0, 0.5, 1.0};
        for (int i = 0; i < 5; i++) {
            assertClose(expectX[i], s.x()[i], 1e-14, "x[" + i + "]");
            assertClose(expectX[i], s.y()[i], 1e-13, "y[" + i + "]");
        }
    }

    @Test
    public void testSamplesLengthOneReturnsMidpoint() {
        Chebfun f = new Chebfun(x -> 3.0 * x + 1.0, new Domain(0.0, 2.0));
        Chebfun.Samples s = f.samples(1);
        assertClose(1.0, s.x()[0], 1e-14, "midpoint x");
        assertClose(4.0, s.y()[0], 1e-13, "midpoint y");
    }

    // -----------------------------------------------------------------
    // fromSamples
    // -----------------------------------------------------------------

    @Test
    public void testFromSamplesInterpolatesLinearExactly() {
        // Linear data: cubic spline is exact.
        double[] x = {0.0, 0.25, 0.5, 0.75, 1.0};
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) y[i] = 2.0 * x[i] + 3.0;
        Chebfun f = Chebfun.fromSamples(x, y, new Domain(0.0, 1.0));
        // Check the spline passes through the knot values.
        for (int i = 0; i < x.length; i++) {
            assertClose(y[i], f.feval(x[i]), 1e-10, "knot " + i);
        }
        // Check off-knot linear values.
        assertClose(2.0 * 0.3 + 3.0, f.feval(0.3), 1e-10, "linear @ 0.3");
        assertClose(2.0 * 0.6 + 3.0, f.feval(0.6), 1e-10, "linear @ 0.6");
    }

    @Test
    public void testFromSamplesRecoversSmoothFunctionWellFromDenseGrid() {
        // sin(x) sampled densely on [0, π], reconstructed via spline+Chebfun.
        int n = 21;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * Math.PI / (n - 1);
            y[i] = Math.sin(x[i]);
        }
        Chebfun f = Chebfun.fromSamples(x, y, new Domain(0.0, Math.PI));
        // Cubic-spline error scales O(h^4); at h=π/20 expect ~1e-4 accuracy.
        for (double xq : new double[]{0.2, 1.0, 1.5, 2.5}) {
            assertClose(Math.sin(xq), f.feval(xq), 5e-4, "sin(x) @ " + xq);
        }
    }

    @Test
    public void testFromSamplesRejectsUnsortedX() {
        double[] x = {0.0, 0.5, 0.3, 1.0};
        double[] y = {0.0, 1.0, 2.0, 3.0};
        try {
            Chebfun.fromSamples(x, y, new Domain(0.0, 1.0));
            throw new AssertionError("expected unsorted x to throw");
        } catch (IllegalArgumentException expected) {
            // OK
        }
    }
}
