package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

/**
 * Initial-value problems via Chebyshev spectral collocation. Each test picks
 * an IVP with a known closed-form solution and checks agreement.
 */
public class IvpTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testLinearExponentialDecay() {
        // u' = -u, u(0) = 1, x ∈ [0, 2] -> u(x) = e^{-x}.
        Domain d = new Domain(0.0, 2.0);
        Chebfun u = Ivp.solve((x, uv) -> -uv, d, 1.0);
        for (double x : new double[]{0.1, 0.5, 1.0, 1.5, 2.0}) {
            assertClose(Math.exp(-x), u.feval(x), 1e-9, "e^{-x} @ " + x);
        }
    }

    @Test
    public void testLinearWithForcing() {
        // u' + u = x, u(0) = 0, x ∈ [0, 1].
        // Exact: u(x) = x - 1 + e^{-x}.
        Domain d = new Domain(0.0, 1.0);
        Chebfun u = Ivp.solve((x, uv) -> x - uv, d, 0.0);
        for (double x : new double[]{0.1, 0.4, 0.7, 1.0}) {
            double want = x - 1.0 + Math.exp(-x);
            assertClose(want, u.feval(x), 1e-9, "x-1+e^{-x} @ " + x);
        }
    }

    @Test
    public void testLogisticGrowthNonlinear() {
        // u' = u (1 - u), u(0) = 0.1, x ∈ [0, 5].
        // Exact: u(x) = 1 / (1 + 9 e^{-x}).
        Domain d = new Domain(0.0, 5.0);
        Chebfun u = Ivp.solve((x, uv) -> uv * (1.0 - uv), d, 0.1);
        for (double x : new double[]{0.5, 1.0, 2.0, 3.0, 4.5}) {
            double want = 1.0 / (1.0 + 9.0 * Math.exp(-x));
            assertClose(want, u.feval(x), 1e-7, "logistic @ " + x);
        }
    }

    @Test
    public void testInitialConditionExact() {
        // The IC row is enforced exactly; verify u(a) = u0 to machine precision.
        Domain d = new Domain(1.0, 3.0);
        double u0 = 0.42;
        Chebfun u = Ivp.solve((x, uv) -> Math.sin(x * uv), d, u0);
        assertClose(u0, u.feval(1.0), 1e-10, "u(a) exact");
    }

    @Test
    public void testAnalyticPartialMatchesFiniteDifference() {
        // Same problem solved with an analytic ∂f/∂u and with the FD default —
        // solutions should be essentially identical (within adaptive-loop tol).
        Domain d = new Domain(0.0, 1.0);
        Chebfun uFD = Ivp.solve((x, uv) -> uv * uv, d, 0.5);
        Chebfun uAn = Ivp.solve(new Ivp.Residual() {
            @Override public double f(double x, double u) { return u * u; }
            @Override public double dU(double x, double u) { return 2.0 * u; }
        }, d, 0.5);
        for (double x : new double[]{0.2, 0.5, 0.8}) {
            assertClose(uFD.feval(x), uAn.feval(x), 1e-8, "FD vs analytic @ " + x);
        }
    }
}
