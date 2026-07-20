package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.Quasimatrix;
import com.marmanis.chebfun4j.Quasimatrix.Algorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * QR and SVD decompositions of a Quasimatrix. Monomial basis demo.
 */
public class QrSvdDemo {
    private static final Logger logger = LoggerFactory.getLogger(QrSvdDemo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: Quasimatrix QR and SVD demos");
        logger.info("=======================================");

        int n = 5;
        Domain d = new Domain(-1.0, 1.0);
        Chebfun[] cols = new Chebfun[n];
        for (int k = 0; k < n; k++) {
            final int power = k;
            cols[k] = new Chebfun(x -> Math.pow(x, power), d);
        }
        Quasimatrix A = new Quasimatrix(cols);

        logger.info(String.format("Quasimatrix A = [1, x, x^2, ..., x^%d] on [-1, 1]", n - 1));

        Quasimatrix.Qr mgs = A.qr(Algorithm.MODIFIED_GRAM_SCHMIDT);
        Quasimatrix.Qr hh  = A.qr(Algorithm.HOUSEHOLDER);
        logger.info("Q'Q on the diagonal (should be 1.0 for orthonormal columns):");
        logger.info(String.format("  MGS Q:        %s", diag(mgs.Q(), n)));
        logger.info(String.format("  Householder Q: %s", diag(hh.Q(), n)));

        double[][] R = mgs.R();
        logger.info("R = Q^T A (upper triangular):");
        for (int i = 0; i < n; i++) {
            StringBuilder row = new StringBuilder("  ");
            for (int j = 0; j < n; j++) row.append(String.format("%9.4f ", R[i][j]));
            logger.info(row.toString());
        }

        Quasimatrix.Svd svd = A.svd();
        logger.info("Singular values sigma_i:");
        double[] sigma = svd.sigma();
        for (int i = 0; i < sigma.length; i++) {
            logger.info(String.format("  sigma_%d = %.10f", i, sigma[i]));
        }
        logger.info(String.format("Condition number of A (sigma_0 / sigma_%d) = %.4f",
                          n - 1, sigma[0] / sigma[sigma.length - 1]));

        double[][] Vt = svd.Vt();
        double maxErr = 0.0;
        double[] xs = {-0.9, -0.4, 0.0, 0.3, 0.7};
        for (int col = 0; col < n; col++) {
            for (double x : xs) {
                double got = 0.0;
                for (int k = 0; k < n; k++) got += svd.U().get(k).feval(x) * sigma[k] * Vt[k][col];
                maxErr = Math.max(maxErr, Math.abs(got - A.get(col).feval(x)));
            }
        }
        logger.info(String.format("max A = U Σ V^T reconstruction error: %.3e", maxErr));
    }

    private static String diag(Quasimatrix Q, int n) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.12f", Q.innerProduct(i, i)));
        }
        return sb.append(']').toString();
    }
}
