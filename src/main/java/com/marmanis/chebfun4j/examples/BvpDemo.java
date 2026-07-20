package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Chebop;
import com.marmanis.chebfun4j.Domain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Solves the linear ODE boundary-value problem
 * <p>
 *   {@code u''(x) - u(x) = 0}, &nbsp; {@code u(0) = 1}, &nbsp; {@code u(1) = 1/e}
 * <p>
 * on {@code [0, 1]}. Closed-form solution: {@code u(x) = e^{-x}}. The demo
 * prints a few pointwise values and the max error against the analytic
 * solution — well below spectral-collocation's usual noise floor.
 */
public class BvpDemo {
    private static final Logger logger = LoggerFactory.getLogger(BvpDemo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: linear ODE BVP demo");
        logger.info("==============================");

        Domain d = new Domain(0.0, 1.0);
        // u'' - u = 0 as (-1) * u + 1 * u''.
        Chebop L = Chebop.constantCoefficients(d, -1.0, 0.0, 1.0);
        Chebfun rhs = Chebfun.constant(0.0, d);
        Chebfun u = L.solve(rhs, 1.0, 1.0 / Math.E);

        logger.info("u'' - u = 0 on [0, 1],  u(0) = 1,  u(1) = 1/e");
        logger.info("  chebfun length = {}", u.length());
        double maxErr = 0.0;
        for (double x : new double[]{0.1, 0.25, 0.5, 0.75, 0.9}) {
            double want = Math.exp(-x);
            double got  = u.feval(x);
            maxErr = Math.max(maxErr, Math.abs(got - want));
            logger.info(String.format("  u(%.2f) = %.15f  (analytic %.15f)", x, got, want));
        }
        logger.info(String.format("  max error on probe set: %.3e", maxErr));
    }
}
