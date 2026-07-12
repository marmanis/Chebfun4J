package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.Quasimatrix;
import com.marmanis.chebfun4j.Quasimatrix.Algorithm;

/**
 * QR and SVD decompositions of a Quasimatrix. Uses the classical
 * example of the monomial basis {@code {1, x, x^2, ..., x^{n-1}}} on
 * {@code [-1, 1]}: MGS and Householder QR give orthonormal Q columns
 * that are (up to sign) the Legendre polynomials, and the SVD reveals
 * the "singular values" of the raw monomial basis.
 */
public class QrSvdDemo {

    public static void main(String[] args) {
        System.out.println("chebfun4j: Quasimatrix QR and SVD demos");
        System.out.println("=======================================");
        System.out.println();

        int n = 5;
        Domain d = new Domain(-1.0, 1.0);
        Chebfun[] cols = new Chebfun[n];
        for (int k = 0; k < n; k++) {
            final int power = k;
            cols[k] = new Chebfun(x -> Math.pow(x, power), d);
        }
        Quasimatrix A = new Quasimatrix(cols);

        System.out.printf("Quasimatrix A = [1, x, x^2, ..., x^%d] on [-1, 1]%n%n", n - 1);

        Quasimatrix.Qr mgs = A.qr(Algorithm.MODIFIED_GRAM_SCHMIDT);
        Quasimatrix.Qr hh  = A.qr(Algorithm.HOUSEHOLDER);
        System.out.println("Q'Q on the diagonal (should be 1.0 for orthonormal columns):");
        System.out.printf("  MGS Q:        %s%n", diag(mgs.Q(), n));
        System.out.printf("  Householder Q: %s%n", diag(hh.Q(), n));
        System.out.println();

        double[][] R = mgs.R();
        System.out.println("R = Q^T A (upper triangular):");
        for (int i = 0; i < n; i++) {
            StringBuilder row = new StringBuilder("  ");
            for (int j = 0; j < n; j++) row.append(String.format("%9.4f ", R[i][j]));
            System.out.println(row);
        }
        System.out.println();

        Quasimatrix.Svd svd = A.svd();
        System.out.println("Singular values sigma_i:");
        double[] sigma = svd.sigma();
        for (int i = 0; i < sigma.length; i++) {
            System.out.printf("  sigma_%d = %.10f%n", i, sigma[i]);
        }
        System.out.println();
        System.out.printf("Condition number of A (sigma_0 / sigma_%d) = %.4f%n",
                          n - 1, sigma[0] / sigma[sigma.length - 1]);

        // Sanity: pointwise reconstruction A = U Σ V^T.
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
        System.out.printf("max A = U Σ V^T reconstruction error: %.3e%n", maxErr);
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
