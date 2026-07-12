package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Chebop;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.Quasimatrix;

/**
 * The classical infinite-well eigenvalue problem:
 *   -u''(x) = lambda u(x)  on  [0, pi],  u(0) = u(pi) = 0.
 *
 * <p>Analytic eigenvalues: {@code lambda_n = n^2}. Analytic eigenfunctions:
 * {@code sqrt(2/pi) sin(n x)}. This example prints the first six numerical
 * eigenvalues alongside the analytic values, and a couple of pointwise
 * values of the first eigenfunction.
 */
public class EigenvalueDemo {
    public static void main(String[] args) {
        System.out.println("chebfun4j: Sturm-Liouville eigenvalue demo");
        System.out.println("==========================================");

        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.zero(d).plus(2, -1.0);          // L u = -u''
        Chebop.Eigs eigs = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 6);

        System.out.println("Eigenvalues of -u'' on [0, pi] with u(0) = u(pi) = 0:");
        System.out.printf("%3s %20s %20s %15s%n", "n", "numerical lambda", "analytic n^2", "abs error");
        for (int i = 0; i < eigs.eigenvalues().length; i++) {
            int n = i + 1;
            double got = eigs.eigenvalues()[i];
            double want = (double) n * n;
            System.out.printf("%3d %20.10f %20.10f %15.2e%n", n, got, want, Math.abs(got - want));
        }
        System.out.println();

        Quasimatrix V = eigs.eigenfunctions();
        Chebfun v1 = V.get(0);
        System.out.println("First eigenfunction v_1 (should be proportional to sin(x)):");
        System.out.printf("  ||v_1||_2 = %.6f (should be 1.0)%n", v1.norm2());
        for (double x : new double[]{Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2}) {
            double got = v1.feval(x);
            double want = Math.sqrt(2.0 / Math.PI) * Math.sin(x);
            System.out.printf("  v_1(%.4f) = %+.10f  (analytic sqrt(2/pi) sin(x) = %+.10f)%n",
                x, got, want);
        }
    }
}
