package com.marmanis.chebfun4j;

import org.junit.jupiter.api.Test;

/**
 * SVD of a Quasimatrix: orthonormal U columns, orthonormal V rows,
 * descending sigma, and A = U Σ V^T reconstruction.
 */
public class QuasimatrixSvdTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    private static Quasimatrix polynomials(Domain d, int n) {
        Chebfun[] cols = new Chebfun[n];
        for (int k = 0; k < n; k++) {
            final int power = k;
            cols[k] = new Chebfun(x -> Math.pow(x, power), d);
        }
        return new Quasimatrix(cols);
    }

    @Test
    public void testSigmaDescendingAndPositive() {
        Quasimatrix A = polynomials(new Domain(-1.0, 1.0), 5);
        Quasimatrix.Svd svd = A.svd();
        double[] s = svd.sigma();
        for (int i = 0; i < s.length; i++) {
            if (s[i] < -1e-10) throw new AssertionError("sigma[" + i + "] = " + s[i] + " < 0");
        }
        for (int i = 0; i < s.length - 1; i++) {
            if (s[i] < s[i + 1] - 1e-10) {
                throw new AssertionError("sigma not sorted at " + i);
            }
        }
    }

    @Test
    public void testUColumnsOrthonormal() {
        Quasimatrix A = polynomials(new Domain(-1.0, 1.0), 4);
        Quasimatrix.Svd svd = A.svd();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double ip = svd.U().innerProduct(i, j);
                double want = (i == j) ? 1.0 : 0.0;
                assertClose(want, ip, 1e-9, "U'U[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testVtRowsOrthonormal() {
        Quasimatrix A = polynomials(new Domain(-1.0, 1.0), 4);
        Quasimatrix.Svd svd = A.svd();
        double[][] Vt = svd.Vt();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                double s = 0.0;
                for (int p = 0; p < 4; p++) s += Vt[i][p] * Vt[j][p];
                double want = (i == j) ? 1.0 : 0.0;
                assertClose(want, s, 1e-10, "VtVt'[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testReconstruction() {
        Domain d = new Domain(-1.0, 1.0);
        Quasimatrix A = polynomials(d, 4);
        Quasimatrix.Svd svd = A.svd();
        double[] sigma = svd.sigma();
        double[][] Vt = svd.Vt();
        double[] xs = {-0.7, -0.3, 0.0, 0.4, 0.8};
        for (int col = 0; col < 4; col++) {
            for (double x : xs) {
                double reconstructed = 0.0;
                for (int k = 0; k < 4; k++) {
                    reconstructed += svd.U().get(k).feval(x) * sigma[k] * Vt[k][col];
                }
                double want = A.get(col).feval(x);
                assertClose(want, reconstructed, 1e-8, "A=UΣVt col=" + col + " @ " + x);
            }
        }
    }
}
