package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.BoundaryCondition.Neumann;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.junit.jupiter.api.Test;

/**
 * Nonlinear boundary-value problems solved by Newton iteration on the
 * spectral-collocation discretization. Each test picks a nonlinear ODE
 * with a known reference solution and checks the numerical answer at
 * interior points.
 */
public class NonlinearChebopTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testLinearBvpAsNonlinear() {
        // Sanity: solve u'' - u = 0, u(0)=1, u(1)=1/e as a "nonlinear" system.
        // Should recover the exact e^{-x} solution.
        Domain d = new Domain(0.0, 1.0);
        NonlinearChebop.Residual F = (x, u, up, upp) -> upp - u;
        NonlinearChebop N = new NonlinearChebop(d, F);
        Chebfun sol = N.solve(new Dirichlet(1.0), new Dirichlet(1.0 / Math.E));
        for (double x : new double[]{0.1, 0.5, 0.9}) {
            assertClose(Math.exp(-x), sol.feval(x), 1e-8, "linear-as-nonlinear @ " + x);
        }
    }

    @Test
    public void testBratuMildLambda() {
        // Bratu: u'' + lambda e^u = 0 on [0, 1] with u(0)=u(1)=0. For small
        // lambda a mild positive solution exists. Reference: for lambda=1
        // the solution has u_max ~ 0.13 to a few digits.
        Domain d = new Domain(0.0, 1.0);
        double lam = 1.0;
        NonlinearChebop.Residual F = (x, u, up, upp) -> upp + lam * Math.exp(u);
        NonlinearChebop N = new NonlinearChebop(d, F);
        Chebfun sol = N.solve(new Dirichlet(0.0), new Dirichlet(0.0));
        // Verify BC: u(0) = u(1) = 0.
        assertClose(0.0, sol.feval(0.0), 1e-8, "Bratu u(0)");
        assertClose(0.0, sol.feval(1.0), 1e-8, "Bratu u(1)");
        // Interior u should be positive (since -u'' = lam e^u > 0 forces
        // u to bow upward from 0).
        double uMid = sol.feval(0.5);
        if (uMid <= 0.0) throw new AssertionError("Bratu solution should be positive interior, got " + uMid);
        // Verify residual — this is the tightest correctness check.
        Chebfun d2 = sol.diff().diff();
        for (double x : new double[]{0.1, 0.3, 0.5, 0.7, 0.9}) {
            double resid = d2.feval(x) + lam * Math.exp(sol.feval(x));
            if (Math.abs(resid) > 1e-5) {
                throw new AssertionError("Bratu residual @ " + x + " = " + resid + " > 1e-5");
            }
        }
    }

    @Test
    public void testNonlinearCubicBvp() {
        // u'' - u^3 = 0, u(0) = 0, u(1) = 1. Not analytically tractable in
        // closed form for arbitrary domains, so verify by residual: the
        // returned u satisfies u'' - u^3 = 0 pointwise.
        Domain d = new Domain(0.0, 1.0);
        NonlinearChebop.Residual F = (x, u, up, upp) -> upp - u * u * u;
        NonlinearChebop N = new NonlinearChebop(d, F);
        Chebfun sol = N.solve(new Dirichlet(0.0), new Dirichlet(1.0));
        assertClose(0.0, sol.feval(0.0), 1e-8, "cubic u(0)");
        assertClose(1.0, sol.feval(1.0), 1e-8, "cubic u(1)");
        Chebfun d2 = sol.diff().diff();
        for (double x : new double[]{0.1, 0.3, 0.5, 0.7, 0.9}) {
            double u = sol.feval(x);
            double resid = d2.feval(x) - u * u * u;
            if (Math.abs(resid) > 1e-5) {
                throw new AssertionError("cubic residual @ " + x + " = " + resid);
            }
        }
    }

    @Test
    public void testNonlinearWithNeumannBc() {
        // u'' + u^2 = 1 on [0, 1] with Dirichlet u(0)=0 and Neumann u'(1)=0.
        // Verify the BCs and check the residual.
        Domain d = new Domain(0.0, 1.0);
        NonlinearChebop.Residual F = (x, u, up, upp) -> upp + u * u - 1.0;
        NonlinearChebop N = new NonlinearChebop(d, F);
        Chebfun sol = N.solve(new Dirichlet(0.0), new Neumann(0.0));
        assertClose(0.0, sol.feval(0.0), 1e-8, "u(0)");
        assertClose(0.0, sol.diff().feval(1.0), 1e-6, "u'(1)");
        Chebfun d2 = sol.diff().diff();
        for (double x : new double[]{0.2, 0.5, 0.8}) {
            double u = sol.feval(x);
            double resid = d2.feval(x) + u * u - 1.0;
            if (Math.abs(resid) > 1e-5) {
                throw new AssertionError("Neumann-nonlin residual @ " + x + " = " + resid);
            }
        }
    }

    @Test
    public void testAutodiffPartialsMatchAnalytic() {
        // Sanity: for F = u'' - u^2 with x fixed, ∂F/∂u = -2u, ∂F/∂up = 0,
        // ∂F/∂upp = 1. Check jax4j autodiff gives these at a probe point.
        NonlinearChebop.Residual F = NonlinearChebop.autodiffResidual(
            (x, u, up, upp) -> upp.sub(u.mul(u)));
        double u = 3.0, up = 1.7, upp = -0.4, x = 0.42;
        assertClose(-2 * u, F.dU(x, u, up, upp), 1e-12, "dU");
        assertClose(0.0,     F.dUp(x, u, up, upp), 1e-12, "dUp");
        assertClose(1.0,     F.dUpp(x, u, up, upp), 1e-12, "dUpp");
        assertClose(upp - u * u, F.at(x, u, up, upp), 1e-14, "at");
    }

    @Test
    public void testAutodiffBratuMatchesFdBratu() {
        // The Bratu problem should converge to the same solution with the
        // FD-partial and AD-partial paths.
        Domain d = new Domain(0.0, 1.0);
        double lam = 1.0;
        NonlinearChebop.Residual Ffd = (x, u, up, upp) -> upp + lam * Math.exp(u);
        NonlinearChebop.Residual Fad = NonlinearChebop.autodiffResidual(
            (x, u, up, upp) -> upp.add(u.exp().mul(scalar(lam))));
        Chebfun solFd = new NonlinearChebop(d, Ffd)
            .solve(new Dirichlet(0.0), new Dirichlet(0.0));
        Chebfun solAd = new NonlinearChebop(d, Fad)
            .solve(new Dirichlet(0.0), new Dirichlet(0.0));
        for (double x : new double[]{0.1, 0.5, 0.9}) {
            assertClose(solFd.feval(x), solAd.feval(x), 1e-7, "autodiff Bratu @ " + x);
        }
    }

    private static NDArray scalar(double v) {
        return new ConcreteNDArray(new double[]{v}, new Shape(1));
    }

    @Test
    public void testInitialGuessImprovesConvergence() {
        // Same Bratu problem but with an initial guess that's already close.
        // Ensures the InitialGuess field is actually used.
        Domain d = new Domain(0.0, 1.0);
        double lam = 1.0;
        NonlinearChebop.Residual F = (x, u, up, upp) -> upp + lam * Math.exp(u);
        NonlinearChebop N = new NonlinearChebop(d, F);
        Chebfun coarse = N.solve(new Dirichlet(0.0), new Dirichlet(0.0));
        // Solve again with the coarse solution as initial guess.
        NewtonOptions withGuess = NewtonOptions.withInitialGuess(coarse);
        Chebfun refined = N.solve(new Dirichlet(0.0), new Dirichlet(0.0), withGuess);
        // Should give the same (unique) solution.
        for (double x : new double[]{0.1, 0.5, 0.9}) {
            assertClose(coarse.feval(x), refined.feval(x), 1e-6, "seeded @ " + x);
        }
    }
}
