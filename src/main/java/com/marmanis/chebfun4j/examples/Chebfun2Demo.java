package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Chebfun2;
import com.marmanis.chebfun4j.Rectangle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A short tour of {@link Chebfun2}: adaptive-rank ACA construction,
 * evaluation, double integrals, marginal integrals returning a 1-D
 * chebfun, and partial derivatives.
 */
public class Chebfun2Demo {
    private static final Logger logger = LoggerFactory.getLogger(Chebfun2Demo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: 2-D functions on a rectangle");
        logger.info("=======================================");

        // Rank-1 example: sin(x) cos(y) on [-1, 1]^2.
        Rectangle unit = Rectangle.unit();
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), unit);
        logger.info("f(x, y) = sin(x) cos(y) on [-1, 1]^2");
        logger.info("  rank         = {}", f.rank());
        logger.info(String.format("  f(0.3, 0.4)  = %.15f (analytic %.15f)",
            f.feval(0.3, 0.4), Math.sin(0.3) * Math.cos(0.4)));
        logger.info(String.format("  double int   = %.15f (analytic 0)", f.sum2()));

        // Rank-1 example: e^{x+y}.
        Chebfun2 g = new Chebfun2((x, y) -> Math.exp(x + y), unit);
        double analytic = (Math.E - 1 / Math.E) * (Math.E - 1 / Math.E); // (e - 1/e)^2
        logger.info("g(x, y) = e^{x+y} on [-1, 1]^2");
        logger.info("  rank         = {}", g.rank());
        logger.info(String.format("  double int   = %.15f (analytic (e-1/e)^2 = %.15f)",
            g.sum2(), analytic));

        // Higher-rank example: 1 / (1 + x^2 + y^2).
        Chebfun2 h = new Chebfun2((x, y) -> 1.0 / (1.0 + x * x + y * y), unit);
        logger.info("h(x, y) = 1 / (1 + x^2 + y^2) on [-1, 1]^2");
        logger.info("  rank         = {} (adaptively chosen)", h.rank());
        logger.info(String.format("  h(0, 0)      = %.15f (analytic 1.0)", h.feval(0.0, 0.0)));
        logger.info(String.format("  h(0.5, 0.5)  = %.15f (analytic %.15f)",
            h.feval(0.5, 0.5), 1.0 / (1.0 + 0.25 + 0.25)));

        // Marginal integral: integrate g over x, leaving a 1-D function of y.
        Chebfun gy = g.sum(0);
        logger.info("integral_{-1}^{1} e^{x+y} dx as a chebfun in y");
        double scale = Math.E - 1 / Math.E;
        for (double y : new double[]{-0.5, 0.0, 0.5}) {
            logger.info(String.format("  gy(%.2f) = %.15f  (analytic %.15f)",
                y, gy.feval(y), scale * Math.exp(y)));
        }

        // Partial derivatives.
        Chebfun2 fx = f.partialX();
        Chebfun2 fy = f.partialY();
        logger.info("Partial derivatives of sin(x) cos(y):");
        logger.info(String.format("  df/dx(0.3, 0.4) = %.15f (analytic cos(0.3) cos(0.4) = %.15f)",
            fx.feval(0.3, 0.4), Math.cos(0.3) * Math.cos(0.4)));
        logger.info(String.format("  df/dy(0.3, 0.4) = %.15f (analytic -sin(0.3) sin(0.4) = %.15f)",
            fy.feval(0.3, 0.4), -Math.sin(0.3) * Math.sin(0.4)));
    }
}
