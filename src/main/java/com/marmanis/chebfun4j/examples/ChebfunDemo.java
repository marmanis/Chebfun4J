package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;

/**
 * A short tour of chebfun4j: construct a smooth function, evaluate,
 * integrate, differentiate, and find roots and extrema — the same handful
 * of one-liners the MATLAB chebfun README opens with.
 *
 * <p>Run with:
 * <pre>
 *   mvn exec:java -Dexec.mainClass=com.marmanis.chebfun4j.examples.ChebfunDemo
 * </pre>
 */
public class ChebfunDemo {
    public static void main(String[] args) {
        System.out.println("chebfun4j demo");
        System.out.println("==============");

        // Example 1: adaptive construction, evaluation, integration.
        Chebfun f = new Chebfun(x -> Math.exp(Math.sin(x)), new Domain(0.0, 2 * Math.PI));
        System.out.printf("f(x) = e^{sin x} on [0, 2 pi]%n");
        System.out.printf("  length()       = %d Chebyshev coefficients%n", f.length());
        System.out.printf("  f(1.0)         = %.15f%n", f.feval(1.0));
        System.out.printf("  integral       = %.15f  (analytic: 2 pi I_0(1) = %.15f)%n",
                          f.sum(), 2 * Math.PI * besselI0(1.0));
        System.out.println();

        // Example 2: derivative and roots.
        Chebfun g = new Chebfun(x -> x * Math.cos(3 * x), new Domain(-1.0, 1.0));
        Chebfun gPrime = g.diff();
        double[] roots = g.roots();
        System.out.printf("g(x) = x cos(3x) on [-1, 1]%n");
        System.out.printf("  length()       = %d%n", g.length());
        System.out.print("  roots           = [");
        for (int i = 0; i < roots.length; i++) {
            System.out.printf("%s%.9f", i == 0 ? "" : ", ", roots[i]);
        }
        System.out.println("]");
        System.out.printf("  g'(0.5)        = %.15f%n", gPrime.feval(0.5));
        System.out.println();

        // Example 3: extrema and norms.
        Chebfun h = new Chebfun(x -> Math.sin(2 * x) + 0.3 * x, new Domain(-3.0, 3.0));
        Chebfun.Extremum lo = h.min();
        Chebfun.Extremum hi = h.max();
        System.out.printf("h(x) = sin(2x) + 0.3 x on [-3, 3]%n");
        System.out.printf("  min            = %.12f at x = %.12f%n", lo.value(), lo.location());
        System.out.printf("  max            = %.12f at x = %.12f%n", hi.value(), hi.location());
        System.out.printf("  ||h||_1        = %.12f%n", h.norm1());
        System.out.printf("  ||h||_2        = %.12f%n", h.norm2());
        System.out.printf("  ||h||_inf      = %.12f%n", h.normInf());
    }

    /**
     * Modified Bessel function of the first kind, order 0. Series form; a
     * dozen terms is plenty at {@code x = 1}. Included so the demo can
     * print a ground-truth value for {@code integral(e^{sin x})}.
     */
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
