package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Chebfun2;
import com.marmanis.chebfun4j.Rectangle;

/**
 * A short tour of {@link Chebfun2}: adaptive-rank ACA construction,
 * evaluation, double integrals, marginal integrals returning a 1-D
 * chebfun, and partial derivatives.
 */
public class Chebfun2Demo {
    public static void main(String[] args) {
        System.out.println("chebfun4j: 2-D functions on a rectangle");
        System.out.println("=======================================");

        // Rank-1 example: sin(x) cos(y) on [-1, 1]^2.
        Rectangle unit = Rectangle.unit();
        Chebfun2 f = new Chebfun2((x, y) -> Math.sin(x) * Math.cos(y), unit);
        System.out.printf("f(x, y) = sin(x) cos(y) on [-1, 1]^2%n");
        System.out.printf("  rank         = %d%n", f.rank());
        System.out.printf("  f(0.3, 0.4)  = %.15f (analytic %.15f)%n",
            f.feval(0.3, 0.4), Math.sin(0.3) * Math.cos(0.4));
        System.out.printf("  double int   = %.15f (analytic 0)%n", f.sum2());
        System.out.println();

        // Rank-1 example: e^{x+y}.
        Chebfun2 g = new Chebfun2((x, y) -> Math.exp(x + y), unit);
        double analytic = (Math.E - 1 / Math.E) * (Math.E - 1 / Math.E); // (e - 1/e)^2
        System.out.printf("g(x, y) = e^{x+y} on [-1, 1]^2%n");
        System.out.printf("  rank         = %d%n", g.rank());
        System.out.printf("  double int   = %.15f (analytic (e-1/e)^2 = %.15f)%n",
            g.sum2(), analytic);
        System.out.println();

        // Higher-rank example: 1 / (1 + x^2 + y^2).
        Chebfun2 h = new Chebfun2((x, y) -> 1.0 / (1.0 + x * x + y * y), unit);
        System.out.printf("h(x, y) = 1 / (1 + x^2 + y^2) on [-1, 1]^2%n");
        System.out.printf("  rank         = %d (adaptively chosen)%n", h.rank());
        System.out.printf("  h(0, 0)      = %.15f (analytic 1.0)%n", h.feval(0.0, 0.0));
        System.out.printf("  h(0.5, 0.5)  = %.15f (analytic %.15f)%n",
            h.feval(0.5, 0.5), 1.0 / (1.0 + 0.25 + 0.25));
        System.out.println();

        // Marginal integral: integrate g over x, leaving a 1-D function of y.
        Chebfun gy = g.sum(0);
        System.out.printf("integral_{-1}^{1} e^{x+y} dx as a chebfun in y%n");
        double scale = Math.E - 1 / Math.E;
        for (double y : new double[]{-0.5, 0.0, 0.5}) {
            System.out.printf("  gy(%.2f) = %.15f  (analytic %.15f)%n",
                y, gy.feval(y), scale * Math.exp(y));
        }
        System.out.println();

        // Partial derivatives.
        Chebfun2 fx = f.partialX();
        Chebfun2 fy = f.partialY();
        System.out.printf("Partial derivatives of sin(x) cos(y):%n");
        System.out.printf("  df/dx(0.3, 0.4) = %.15f (analytic cos(0.3) cos(0.4) = %.15f)%n",
            fx.feval(0.3, 0.4), Math.cos(0.3) * Math.cos(0.4));
        System.out.printf("  df/dy(0.3, 0.4) = %.15f (analytic -sin(0.3) sin(0.4) = %.15f)%n",
            fy.feval(0.3, 0.4), -Math.sin(0.3) * Math.sin(0.4));
    }
}
