package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Constructs {@code |x|} on {@code [-1, 1]} and demonstrates that the
 * iteration-2 splitting-on constructor puts a breakpoint at the kink.
 */
public class PiecewiseDemo {
    private static final Logger logger = LoggerFactory.getLogger(PiecewiseDemo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: piecewise |x| demo");
        logger.info("=============================");

        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));

        logger.info("f(x) = |x| on [-1, 1]");
        logger.info("  numPieces()   = {} (smooth pieces glued at kinks)", f.numPieces());
        double[] bp = f.breakpoints();
        
        StringBuilder sb = new StringBuilder("  breakpoints   = [");
        for (int i = 0; i < bp.length; i++) {
            sb.append(String.format(i == 0 ? "%.12f" : ", %.12f", bp[i]));
        }
        sb.append("]");
        logger.info(sb.toString());

        logger.info(String.format("  integral      = %.15f (analytic: 1)", f.sum()));
        logger.info(String.format("  ||f||_inf     = %.15f (analytic: 1)", f.normInf()));
        logger.info(String.format("  ||f||_1       = %.15f (analytic: 1)", f.norm1()));

        // Derivative: sign(x) with a jump at 0.
        Chebfun df = f.diff();
        logger.info("d|x|/dx on [-1, 1]");
        logger.info("  numPieces()   = {}", df.numPieces());
        logger.info(String.format("  df(-0.5)      = %.15f (analytic: -1)", df.feval(-0.5)));
        logger.info(String.format("  df(+0.5)      = %.15f (analytic: +1)", df.feval(+0.5)));
    }
}
