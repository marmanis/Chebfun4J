package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.SystemBC.Dirichlet;
import com.marmanis.chebfun4j.SystemBC.Neumann;
import org.junit.jupiter.api.Test;

/**
 * Nonlinear systems of ODE BVPs. Each test picks a system with known
 * closed-form solution or verifies via residual satisfaction.
 */
public class NonlinearSystemTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testLinearSystemAsNonlinearSanity() {
        // The same decoupled linear system solved as a "nonlinear" one.
        //   u_0'' - u_0 = 0, u_0(0)=1, u_0(1)=1/e -> e^{-x}
        //   u_1'' + u_1 = 0, u_1(0)=0, u_1(1)=sin(1) -> sin(x)
        Domain d = new Domain(0.0, 1.0);
        NonlinearSystem.Residual F = new NonlinearSystem.Residual() {
            @Override public int numComponents() { return 2; }
            @Override public double[] at(double x, double[] u, double[] up, double[] upp) {
                return new double[]{ upp[0] - u[0], upp[1] + u[1] };
            }
        };
        NonlinearSystem sys = new NonlinearSystem(d, F);
        SystemBC[] bcA = { new Dirichlet(0, 1.0),           new Dirichlet(1, 0.0) };
        SystemBC[] bcB = { new Dirichlet(0, 1.0 / Math.E),  new Dirichlet(1, Math.sin(1.0)) };
        Chebfun[] u = sys.solve(bcA, bcB);
        for (double x : new double[]{0.2, 0.5, 0.8}) {
            assertClose(Math.exp(-x), u[0].feval(x), 1e-7, "u_0 @ " + x);
            assertClose(Math.sin(x),  u[1].feval(x), 1e-7, "u_1 @ " + x);
        }
    }

    @Test
    public void testCoupledNonlinearSystem() {
        // u_0'' = -u_1^2 ,  u_1'' = -u_0^2   on [0, 1], u_0(0)=u_1(0)=0,
        // u_0(1)=u_1(1) = 1. Symmetric under u_0<->u_1 so at the solution
        // u_0 == u_1. Verify by residual after solve.
        Domain d = new Domain(0.0, 1.0);
        NonlinearSystem.Residual F = new NonlinearSystem.Residual() {
            @Override public int numComponents() { return 2; }
            @Override public double[] at(double x, double[] u, double[] up, double[] upp) {
                return new double[]{ upp[0] + u[1] * u[1], upp[1] + u[0] * u[0] };
            }
        };
        NonlinearSystem sys = new NonlinearSystem(d, F);
        SystemBC[] bcA = { new Dirichlet(0, 0.0), new Dirichlet(1, 0.0) };
        SystemBC[] bcB = { new Dirichlet(0, 1.0), new Dirichlet(1, 1.0) };
        Chebfun[] u = sys.solve(bcA, bcB);
        // Verify BCs.
        assertClose(0.0, u[0].feval(0.0), 1e-6, "u_0(0)");
        assertClose(0.0, u[1].feval(0.0), 1e-6, "u_1(0)");
        assertClose(1.0, u[0].feval(1.0), 1e-6, "u_0(1)");
        assertClose(1.0, u[1].feval(1.0), 1e-6, "u_1(1)");
        // Verify residuals interior.
        Chebfun u0pp = u[0].diff().diff();
        Chebfun u1pp = u[1].diff().diff();
        for (double x : new double[]{0.25, 0.5, 0.75}) {
            double r0 = u0pp.feval(x) + u[1].feval(x) * u[1].feval(x);
            double r1 = u1pp.feval(x) + u[0].feval(x) * u[0].feval(x);
            if (Math.abs(r0) > 1e-5) throw new AssertionError("r0 @ " + x + " = " + r0);
            if (Math.abs(r1) > 1e-5) throw new AssertionError("r1 @ " + x + " = " + r1);
        }
        // By symmetry, u_0 == u_1.
        for (double x : new double[]{0.3, 0.7}) {
            assertClose(u[0].feval(x), u[1].feval(x), 1e-6, "symmetry @ " + x);
        }
    }

    @Test
    public void testMixedBcs() {
        // Same coupled system as above but with Neumann at one endpoint.
        Domain d = new Domain(0.0, 1.0);
        NonlinearSystem.Residual F = new NonlinearSystem.Residual() {
            @Override public int numComponents() { return 2; }
            @Override public double[] at(double x, double[] u, double[] up, double[] upp) {
                return new double[]{ upp[0] + u[1] * u[1], upp[1] + u[0] * u[0] };
            }
        };
        NonlinearSystem sys = new NonlinearSystem(d, F);
        SystemBC[] bcA = { new Dirichlet(0, 0.0), new Neumann(1, 0.0) };
        SystemBC[] bcB = { new Dirichlet(0, 1.0), new Dirichlet(1, 0.5) };
        Chebfun[] u = sys.solve(bcA, bcB);
        assertClose(0.0, u[0].feval(0.0), 1e-6, "u_0(0)");
        assertClose(1.0, u[0].feval(1.0), 1e-6, "u_0(1)");
        assertClose(0.5, u[1].feval(1.0), 1e-6, "u_1(1)");
        // Verify u_1'(0) = 0 (Neumann).
        assertClose(0.0, u[1].diff().feval(0.0), 1e-4, "u_1'(0)");
    }
}
