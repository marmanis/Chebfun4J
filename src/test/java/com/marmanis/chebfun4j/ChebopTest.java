package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

/**
 * Spectral-collocation BVP solves. Each test picks an ODE with a known
 * closed-form solution and checks that {@code Chebop.solve} matches it at
 * a handful of interior points.
 */
public class ChebopTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSimpleHarmonic() {
        // u'' + u = 0 on [0, pi/2] with u(0) = 1, u(pi/2) = 1.
        // Solution: u(x) = cos(x) + sin(x).
        Domain d = new Domain(0.0, Math.PI / 2);
        Chebop L = Chebop.constantCoefficients(d, 1.0, 0.0, 1.0);
        Chebfun rhs = Chebfun.constant(0.0, d);
        Chebfun u = L.solve(rhs, 1.0, 1.0);
        for (double x : new double[]{0.1, 0.5, 1.0, 1.4}) {
            assertClose(Math.cos(x) + Math.sin(x), u.feval(x), 1e-8, "SHO @ " + x);
        }
    }

    @Test
    public void testPoisson() {
        // u'' = -1 on [0, 1] with u(0) = u(1) = 0.
        // Solution: u(x) = x (1 - x) / 2.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, 0.0, 0.0, 1.0);
        Chebfun rhs = Chebfun.constant(-1.0, d);
        Chebfun u = L.solve(rhs, 0.0, 0.0);
        for (double x : new double[]{0.1, 0.25, 0.5, 0.75, 0.9}) {
            assertClose(0.5 * x * (1 - x), u.feval(x), 1e-10, "Poisson @ " + x);
        }
    }

    @Test
    public void testFirstOrderLinear() {
        // u' + u = x on [0, 1] with u(0) = 1, u(1) = 1/e.
        // General: u(x) = x - 1 + C e^{-x}. u(0) = 1 gives C = 2. But then
        // u(1) = 1 - 1 + 2/e = 2/e != 1/e. To force both BCs, we can't; a
        // 1st-order ODE has just one BC. Instead: solve u' + u = x on [0, 1]
        // with just u(0) = 1 by using a Dirichlet-at-both-ends BVP with the
        // consistent value at x=1: pick beta = 2/e - 1 + 1 = 2/e (the true
        // solution at x=1 given u(0)=1). This over-determines but any
        // consistent choice works.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, 1.0, 1.0);
        double alpha = 1.0;                                   // u(0) = 1
        double beta = 2.0 * Math.exp(-1.0);                   // u(1) matches u(x) = x - 1 + 2 e^{-x}
        Chebfun u = L.solve(x -> x, alpha, beta);
        for (double x : new double[]{0.1, 0.4, 0.7}) {
            double want = x - 1 + 2 * Math.exp(-x);
            assertClose(want, u.feval(x), 1e-8, "1st-order @ " + x);
        }
    }

    @Test
    public void testVariableCoefficient() {
        // (1 + x^2) u'' + u = 0 on [-1, 1] with u(-1) = u(1) = 1. No neat
        // closed form; verify the residual instead: build the numerical u,
        // compute L u pointwise, compare to zero.
        Domain d = new Domain(-1.0, 1.0);
        Chebfun onePlusXSq = new Chebfun(x -> 1 + x * x, d);
        Chebop L = Chebop.zero(d).plus(2, onePlusXSq).plus(0, 1.0);
        Chebfun rhs = Chebfun.constant(0.0, d);
        Chebfun u = L.solve(rhs, 1.0, 1.0);
        // Verify residual: (1+x^2) u'' + u should be near zero on the interior.
        Chebfun d2 = u.diff().diff();
        for (double x : new double[]{-0.5, 0.0, 0.5}) {
            double residual = (1 + x * x) * d2.feval(x) + u.feval(x);
            assertClose(0.0, residual, 1e-6, "var-coef residual @ " + x);
        }
    }

    @Test
    public void testBcsAreExact() {
        // The Dirichlet BCs should be reproduced exactly at the endpoints.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, 0.0, 0.0, 1.0);
        Chebfun rhs = Chebfun.constant(-1.0, d);
        Chebfun u = L.solve(rhs, 0.3, 0.7);
        assertClose(0.3, u.feval(0.0), 1e-10, "u(a)");
        assertClose(0.7, u.feval(1.0), 1e-10, "u(b)");
    }

    // -----------------------------------------------------------------
    // Discretization + conditionNumber
    // -----------------------------------------------------------------

    @Test
    public void testDiscretizeSolveMatchesDirectSolve() {
        // u'' - u = 0 on [0, 1] with u(0) = 1, u(1) = 1/e. Solution: e^{-x}.
        // discretize + solve should produce the same Chebfun as L.solve at
        // the corresponding grid size.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebop.Discretization disc = L.discretize(
            new BoundaryCondition.Dirichlet(1.0),
            new BoundaryCondition.Dirichlet(1.0 / Math.E),
            32);
        Chebfun u = disc.solve(x -> 0.0);
        for (double x : new double[]{0.1, 0.3, 0.5, 0.7, 0.9}) {
            assertClose(Math.exp(-x), u.feval(x), 1e-10, "discretize-solved @ " + x);
        }
    }

    @Test
    public void testDiscretizeReusableAcrossMultipleRhs() {
        // Same operator + BCs, three different RHS. Each solve should match
        // a fresh L.solve at the same n.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, 0.0, 0.0, 1.0); // u''
        Chebop.Discretization disc = L.discretize(
            new BoundaryCondition.Dirichlet(0.0),
            new BoundaryCondition.Dirichlet(0.0),
            64);
        // u'' = -pi^2 sin(pi x) -> u(x) = sin(pi x).
        Chebfun uSin = disc.solve(x -> -Math.PI * Math.PI * Math.sin(Math.PI * x));
        assertClose(1.0, uSin.feval(0.5), 1e-10, "sin @ 0.5");
        // u'' = 12 x - 6 -> u(x) = 2x^3 - 3x^2 + x (Dirichlet 0/0).
        Chebfun uCubic = disc.solve(x -> 12 * x - 6);
        assertClose(2 * 0.5 * 0.5 * 0.5 - 3 * 0.5 * 0.5 + 0.5, uCubic.feval(0.5), 1e-10, "cubic @ 0.5");
        // u'' = 2 -> u(x) = x^2 - x (Dirichlet 0/0).
        Chebfun uQuad = disc.solve(x -> 2.0);
        assertClose(-0.25, uQuad.feval(0.5), 1e-10, "quadratic @ 0.5");
    }

    @Test
    public void testConditionNumberIsFiniteAndSensible() {
        // -u'' on [0, π] with Dirichlet(0, 0). Well-posed BVP; κ should be
        // finite, > 1, and grow with n (Chebyshev D² has O(n²) growth).
        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.constantCoefficients(d, 0.0, 0.0, -1.0); // -u''
        BoundaryCondition bcA = new BoundaryCondition.Dirichlet(0.0);
        BoundaryCondition bcB = new BoundaryCondition.Dirichlet(0.0);
        double c32 = L.discretize(bcA, bcB, 32).conditionNumber();
        double c128 = L.discretize(bcA, bcB, 128).conditionNumber();
        if (!Double.isFinite(c32) || c32 <= 1.0) {
            throw new AssertionError("cond at n=32 not finite/>1: " + c32);
        }
        if (!Double.isFinite(c128) || c128 <= c32) {
            throw new AssertionError("cond at n=128 should exceed cond at n=32; got "
                + c32 + " vs " + c128);
        }
    }

    @Test
    public void testDiscretizationReportsCorrectN() {
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, 0.0, 0.0, 1.0);
        Chebop.Discretization disc = L.discretize(
            new BoundaryCondition.Dirichlet(0.0),
            new BoundaryCondition.Dirichlet(0.0),
            64);
        if (disc.n() != 64) {
            throw new AssertionError("expected n=64, got " + disc.n());
        }
    }
}
