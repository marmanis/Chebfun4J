package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A short tour of chebfun4j: construct a smooth function, evaluate,
 * integrate, differentiate, and find roots and extrema — the same handful
 * of one-liners the MATLAB chebfun README opens with.
 */
public class ChebfunDemo {
    private static final Logger logger = LoggerFactory.getLogger(ChebfunDemo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j demo");
        logger.info("==============");

        // Example 1: adaptive construction, evaluation, integration.
        Chebfun f = new Chebfun(x -> Math.exp(Math.sin(x)), new Domain(0.0, 2 * Math.PI));
        logger.info("f(x) = e^{sin x} on [0, 2 pi]");
        logger.info("  length()       = {} Chebyshev coefficients", f.length());
        logger.info(String.format("  f(1.0)         = %.15f", f.feval(1.0)));
        logger.info(String.format("  integral       = %.15f  (analytic: 2 pi I_0(1) = %.15f)",
                          f.sum(), 2 * Math.PI * besselI0(1.0)));

        // Example 2: derivative and roots.
        Chebfun g = new Chebfun(x -> x * Math.cos(3 * x), new Domain(-1.0, 1.0));
        Chebfun gPrime = g.diff();
        double[] roots = g.roots();
        logger.info("g(x) = x cos(3x) on [-1, 1]");
        logger.info("  length()       = {}", g.length());
        
        StringBuilder sb = new StringBuilder("  roots           = [");
        for (int i = 0; i < roots.length; i++) {
            sb.append(String.format(i == 0 ? "%.9f" : ", %.9f", roots[i]));
        }
        sb.append("]");
        logger.info(sb.toString());
        
        logger.info(String.format("  g'(0.5)        = %.15f", gPrime.feval(0.5)));

        // Example 3: extrema and norms.
        Chebfun h = new Chebfun(x -> Math.sin(2 * x) + 0.3 * x, new Domain(-3.0, 3.0));
        Chebfun.Extremum lo = h.min();
        Chebfun.Extremum hi = h.max();
        logger.info("h(x) = sin(2x) + 0.3 x on [-3, 3]");
        logger.info(String.format("  min            = %.12f at x = %.12f", lo.value(), lo.location()));
        logger.info(String.format("  max            = %.12f at x = %.12f", hi.value(), hi.location()));
        logger.info(String.format("  ||h||_1        = %.12f", h.norm1()));
        logger.info(String.format("  ||h||_2        = %.12f", h.norm2()));
        logger.info(String.format("  ||h||_inf      = %.12f", h.normInf()));
    }

    private static double besselI0(double x) {
        double sum = 1.0;
        double term = 1.0;
        double y = x * x / 4.0;
        for (int k = 1; k <= 25; k++) {
            term *= y / (k * k);
            sum += term;
        }
        return sum;
    }
}
