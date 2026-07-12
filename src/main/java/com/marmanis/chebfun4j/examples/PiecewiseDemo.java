package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;

/**
 * Constructs {@code |x|} on {@code [-1, 1]} — a function the smooth
 * (iteration-1) constructor couldn't resolve — and demonstrates that the
 * iteration-2 splitting-on constructor puts a breakpoint at the kink,
 * then integrates and differentiates across pieces.
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -Dexec.mainClass=com.marmanis.chebfun4j.examples.PiecewiseDemo
 * </pre>
 */
public class PiecewiseDemo {
    public static void main(String[] args) {
        System.out.println("chebfun4j: piecewise |x| demo");
        System.out.println("=============================");

        Chebfun f = new Chebfun(Math::abs, new Domain(-1.0, 1.0));

        System.out.printf("f(x) = |x| on [-1, 1]%n");
        System.out.printf("  numPieces()   = %d (smooth pieces glued at kinks)%n", f.numPieces());
        double[] bp = f.breakpoints();
        System.out.printf("  breakpoints   = [");
        for (int i = 0; i < bp.length; i++) {
            System.out.printf("%s%.12f", i == 0 ? "" : ", ", bp[i]);
        }
        System.out.println("]");
        System.out.printf("  integral      = %.15f (analytic: 1)%n", f.sum());
        System.out.printf("  ||f||_inf     = %.15f (analytic: 1)%n", f.normInf());
        System.out.printf("  ||f||_1       = %.15f (analytic: 1)%n", f.norm1());
        System.out.println();

        // Derivative: sign(x) with a jump at 0.
        Chebfun df = f.diff();
        System.out.printf("d|x|/dx on [-1, 1]%n");
        System.out.printf("  numPieces()   = %d%n", df.numPieces());
        System.out.printf("  df(-0.5)      = %.15f (analytic: -1)%n", df.feval(-0.5));
        System.out.printf("  df(+0.5)      = %.15f (analytic: +1)%n", df.feval(+0.5));
    }
}
