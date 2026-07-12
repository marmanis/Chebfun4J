package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.SystemBC.Dirichlet;
import com.marmanis.chebfun4j.SystemBC.Neumann;
import org.junit.jupiter.api.Test;

/**
 * Coupled linear BVPs. Each test picks a system with known analytic
 * solution and verifies the numerical answer at interior points.
 */
public class LinearBlockChebopTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testDecoupledDiagonalSystem() {
        // Two independent scalar problems solved as one 2-component system.
        //   u_0'' - u_0 = 0,  u_0(0) = 1, u_0(1) = 1/e  -> u_0 = e^{-x}
        //   u_1'' + u_1 = 0,  u_1(0) = 0, u_1(1) = sin(1) -> u_1 = sin(x)
        Domain d = new Domain(0.0, 1.0);
        LinearBlockChebop L = LinearBlockChebop.zero(d, 2)
            .term(0, 0, 2, 1.0).term(0, 0, 0, -1.0)   // u_0'' - u_0
            .term(1, 1, 2, 1.0).term(1, 1, 0, 1.0);   // u_1'' + u_1
        Chebfun[] rhs = { Chebfun.constant(0.0, d), Chebfun.constant(0.0, d) };
        SystemBC[] bcA = { new Dirichlet(0, 1.0),         new Dirichlet(1, 0.0) };
        SystemBC[] bcB = { new Dirichlet(0, 1.0 / Math.E), new Dirichlet(1, Math.sin(1.0)) };
        Chebfun[] u = L.solve(rhs, bcA, bcB);
        for (double x : new double[]{0.1, 0.5, 0.9}) {
            assertClose(Math.exp(-x), u[0].feval(x), 1e-8, "u_0 @ " + x);
            assertClose(Math.sin(x),  u[1].feval(x), 1e-8, "u_1 @ " + x);
        }
    }

    @Test
    public void testCoupledFirstOrderSystem() {
        // Convert u'' = -u into a 1st-order system:
        //   u_0' = u_1
        //   u_1' = -u_0
        // On [0, pi/2] with u_0(0) = 0, u_0(pi/2) = 1. Solution: u_0 = sin(x),
        // u_1 = cos(x). Since it's 1st-order in two variables (total order 2),
        // we need 2 BCs (one per endpoint here).
        Domain d = new Domain(0.0, Math.PI / 2);
        LinearBlockChebop L = LinearBlockChebop.zero(d, 2)
            .term(0, 0, 1, 1.0).term(0, 1, 0, -1.0)    // u_0' - u_1 = 0
            .term(1, 0, 0, 1.0).term(1, 1, 1, 1.0);    // u_0 + u_1' = 0
        Chebfun[] rhs = { Chebfun.constant(0.0, d), Chebfun.constant(0.0, d) };
        SystemBC[] bcA = { new Dirichlet(0, 0.0) };
        SystemBC[] bcB = { new Dirichlet(0, 1.0) };
        Chebfun[] u = L.solve(rhs, bcA, bcB);
        for (double x : new double[]{0.2, 0.5, 1.0, 1.3}) {
            assertClose(Math.sin(x), u[0].feval(x), 1e-8, "sin @ " + x);
            assertClose(Math.cos(x), u[1].feval(x), 1e-8, "cos @ " + x);
        }
    }

    @Test
    public void testMixedBcTypes() {
        // Same decoupled diagonal system but with a Neumann BC on the second
        // component.  u_1'' - u_1 = 0, u_1'(0) = 1, u_1(1) = e.
        // General solution: u_1 = A e^x + B e^{-x}. u_1'(0) = A - B = 1;
        // u_1(1) = A e + B/e = e. Solve: A = 1 + B, (1+B) e + B/e = e,
        // e + B e + B/e = e -> B(e + 1/e) = 0 -> B = 0, A = 1. So u_1 = e^x.
        Domain d = new Domain(0.0, 1.0);
        LinearBlockChebop L = LinearBlockChebop.zero(d, 2)
            .term(0, 0, 2, 1.0).term(0, 0, 0, -1.0)   // u_0'' - u_0
            .term(1, 1, 2, 1.0).term(1, 1, 0, -1.0);  // u_1'' - u_1
        Chebfun[] rhs = { Chebfun.constant(0.0, d), Chebfun.constant(0.0, d) };
        SystemBC[] bcA = { new Dirichlet(0, 1.0),         new Neumann(1, 1.0) };
        SystemBC[] bcB = { new Dirichlet(0, 1.0 / Math.E), new Dirichlet(1, Math.E) };
        Chebfun[] u = L.solve(rhs, bcA, bcB);
        for (double x : new double[]{0.1, 0.5, 0.9}) {
            assertClose(Math.exp(-x), u[0].feval(x), 1e-8, "u_0 @ " + x);
            assertClose(Math.exp(x),  u[1].feval(x), 1e-8, "u_1 @ " + x);
        }
    }

    @Test
    public void testInvalidComponentIndexThrows() {
        Domain d = new Domain(0.0, 1.0);
        try {
            LinearBlockChebop.zero(d, 2).term(2, 0, 0, 1.0);
            throw new AssertionError("expected exception");
        } catch (IllegalArgumentException expected) {}
    }

    @Test
    public void testInvalidComponentCountThrows() {
        try {
            LinearBlockChebop.zero(new Domain(0.0, 1.0), 0);
            throw new AssertionError("expected exception");
        } catch (IllegalArgumentException expected) {}
    }
}
