package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.BoundaryCondition.Neumann;
import com.marmanis.chebfun4j.BoundaryCondition.Robin;
import org.junit.jupiter.api.Test;

/**
 * BVPs using Neumann and Robin boundary conditions. Each test picks an
 * ODE with a known closed-form solution and checks the numerical answer
 * against it at interior points, and verifies that the BC itself is
 * reproduced at the endpoint (the derivative for Neumann, the linear
 * combination for Robin).
 */
public class ChebopBoundaryConditionTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testNeumannAtLeftDirichletAtRight() {
        // u'' = 0 on [0, 1] with u'(0) = 0 and u(1) = 5. Solution: u = 5
        // (constant, since zero slope + constant right-endpoint value).
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, 0.0, 0.0, 1.0);
        Chebfun u = L.solve(x -> 0.0, new Neumann(0.0), new Dirichlet(5.0));
        for (double x : new double[]{0.0, 0.3, 0.7, 1.0}) {
            assertClose(5.0, u.feval(x), 1e-10, "u @ " + x);
        }
    }

    @Test
    public void testNeumannBothEnds() {
        // u'' - u = 0 on [0, 1] with u'(0) = 1 and u'(1) = e. Solution:
        // u = e^x (uniquely: u'' - u = 0 with these Neumann values pins
        // the two constants of the general solution A e^x + B e^{-x}).
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebfun u = L.solve(x -> 0.0, new Neumann(1.0), new Neumann(Math.E));
        for (double x : new double[]{0.0, 0.25, 0.5, 0.75, 1.0}) {
            assertClose(Math.exp(x), u.feval(x), 1e-10, "u @ " + x);
        }
        // Verify Neumann BCs are reproduced by the numerical derivative.
        Chebfun up = u.diff();
        assertClose(1.0, up.feval(0.0), 1e-9, "u'(0) = 1");
        assertClose(Math.E, up.feval(1.0), 1e-9, "u'(1) = e");
    }

    @Test
    public void testDirichletAtLeftNeumannAtRight() {
        // u'' - u = 0 on [0, 1] with u(0) = 1 and u'(1) = -1/e.
        // Solution: u = e^{-x}. u(0)=1, u'(x) = -e^{-x}, u'(1) = -1/e.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebfun u = L.solve(x -> 0.0, new Dirichlet(1.0), new Neumann(-1.0 / Math.E));
        for (double x : new double[]{0.0, 0.25, 0.5, 0.75, 1.0}) {
            assertClose(Math.exp(-x), u.feval(x), 1e-10, "u @ " + x);
        }
    }

    @Test
    public void testRobinBoundary() {
        // u'' - u = 0 on [0, 1]. Robin at x=0: u(0) - u'(0) = 2 (i.e.
        // 1*u + (-1)*u' = 2). Dirichlet at x=1: u(1) = e.
        // Solution family: u = A e^x + B e^{-x}.
        // u(0) - u'(0) = A + B - (A - B) = 2B = 2 -> B = 1.
        // u(1) = A e + 1/e = e -> A = 1 - 1/e^2 ~ 0.8647.
        double A = 1.0 - 1.0 / (Math.E * Math.E);
        double B = 1.0;
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebfun u = L.solve(x -> 0.0, new Robin(1.0, -1.0, 2.0), new Dirichlet(Math.E));
        for (double x : new double[]{0.1, 0.3, 0.5, 0.7, 0.9}) {
            double want = A * Math.exp(x) + B * Math.exp(-x);
            assertClose(want, u.feval(x), 1e-9, "robin u @ " + x);
        }
        // Verify Robin combination at x=0 is 2 (to solver tolerance).
        Chebfun up = u.diff();
        double lhs = u.feval(0.0) - up.feval(0.0);
        assertClose(2.0, lhs, 1e-8, "u(0) - u'(0) = 2 (Robin BC)");
    }

    @Test
    public void testBackwardCompatibleDoubleOverloadStillWorks() {
        // Old signature `solve(rhs, alpha, beta)` should still route to
        // Dirichlet BCs on both sides — regression protection for
        // iteration 2 users.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebfun u = L.solve(Chebfun.constant(0.0, d), 1.0, 1.0 / Math.E);
        for (double x : new double[]{0.1, 0.5, 0.9}) {
            assertClose(Math.exp(-x), u.feval(x), 1e-10, "legacy path @ " + x);
        }
    }
}
