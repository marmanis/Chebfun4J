package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Chebop;
import com.marmanis.chebfun4j.Domain;

/**
 * Solves the linear ODE boundary-value problem
 * <p>
 *   {@code u''(x) - u(x) = 0}, &nbsp; {@code u(0) = 1}, &nbsp; {@code u(1) = 1/e}
 * <p>
 * on {@code [0, 1]}. Closed-form solution: {@code u(x) = e^{-x}}. The demo
 * prints a few pointwise values and the max error against the analytic
 * solution — well below spectral-collocation's usual noise floor.
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -Dexec.mainClass=com.marmanis.chebfun4j.examples.BvpDemo
 * </pre>
 */
public class BvpDemo {
    public static void main(String[] args) {
        System.out.println("chebfun4j: linear ODE BVP demo");
        System.out.println("==============================");

        Domain d = new Domain(0.0, 1.0);
        // u'' - u = 0 as (-1) * u + 1 * u''.
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebfun rhs = Chebfun.constant(0.0, d);
        Chebfun u = L.solve(rhs, 1.0, 1.0 / Math.E);

        System.out.printf("u'' - u = 0 on [0, 1],  u(0) = 1,  u(1) = 1/e%n");
        System.out.printf("  chebfun length = %d%n", u.length());
        double maxErr = 0.0;
        for (double x : new double[]{0.1, 0.25, 0.5, 0.75, 0.9}) {
            double want = Math.exp(-x);
            double got  = u.feval(x);
            maxErr = Math.max(maxErr, Math.abs(got - want));
            System.out.printf("  u(%.2f) = %.15f  (analytic %.15f)%n", x, got, want);
        }
        System.out.printf("  max error on probe set: %.3e%n", maxErr);
    }
}
