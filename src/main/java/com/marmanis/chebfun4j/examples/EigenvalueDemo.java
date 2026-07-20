package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Chebop;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.Quasimatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The classical infinite-well eigenvalue problem:
 *   -u''(x) = lambda u(x)  on  [0, pi],  u(0) = u(pi) = 0.
 */
public class EigenvalueDemo {
    private static final Logger logger = LoggerFactory.getLogger(EigenvalueDemo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: Sturm-Liouville eigenvalue demo");
        logger.info("==========================================");

        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.zero(d).plus(2, -1.0);          // L u = -u''
        Chebop.Eigs eigs = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 6);

        logger.info("Eigenvalues of -u'' on [0, pi] with u(0) = u(pi) = 0:");
        logger.info(String.format("%3s %20s %20s %15s", "n", "numerical lambda", "analytic n^2", "abs error"));
        for (int i = 0; i < eigs.eigenvalues().length; i++) {
            int n = i + 1;
            double got = eigs.eigenvalues()[i];
            double want = (double) n * n;
            logger.info(String.format("%3d %20.10f %20.10f %15.2e", n, got, want, Math.abs(got - want)));
        }

        Quasimatrix V = eigs.eigenfunctions();
        Chebfun v1 = V.get(0);
        logger.info("First eigenfunction v_1 (should be proportional to sin(x)):");
        logger.info(String.format("  ||v_1||_2 = %.6f (should be 1.0)", v1.norm2()));
        for (double x : new double[]{Math.PI / 6, Math.PI / 4, Math.PI / 3, Math.PI / 2}) {
            double got = v1.feval(x);
            double want = Math.sqrt(2.0 / Math.PI) * Math.sin(x);
            logger.info(String.format("  v_1(%.4f) = %+.10f  (analytic sqrt(2/pi) sin(x) = %+.10f)",
                x, got, want));
        }
    }
}
