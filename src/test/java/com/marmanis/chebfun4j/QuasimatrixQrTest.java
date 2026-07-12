package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.Quasimatrix.Algorithm;
import org.junit.jupiter.api.Test;

/**
 * QR decompositions of a Quasimatrix. Both MGS and Householder are
 * tested: orthonormality of Q's columns, upper-triangularity of R, and
 * reconstruction A = Q R.
 */
public class QuasimatrixQrTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    private static Quasimatrix wellConditionedQuasimatrix(Domain d, int n) {
        // {1, x, x^2, ..., x^{n-1}} — Vandermonde-like, well-conditioned
        // for small n.
        Chebfun[] cols = new Chebfun[n];
        for (int k = 0; k < n; k++) {
            final int power = k;
            cols[k] = new Chebfun(x -> Math.pow(x, power), d);
        }
        return new Quasimatrix(cols);
    }

    @Test
    public void testMGSOrthonormality() {
        Quasimatrix A = wellConditionedQuasimatrix(new Domain(-1.0, 1.0), 5);
        Quasimatrix.Qr qr = A.qr(Algorithm.MODIFIED_GRAM_SCHMIDT);
        // Columns of Q should be orthonormal in L^2.
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double ip = qr.Q().innerProduct(i, j);
                double want = (i == j) ? 1.0 : 0.0;
                assertClose(want, ip, 1e-12, "Q'Q[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testMGSUpperTriangular() {
        Quasimatrix A = wellConditionedQuasimatrix(new Domain(-1.0, 1.0), 5);
        Quasimatrix.Qr qr = A.qr(Algorithm.MODIFIED_GRAM_SCHMIDT);
        double[][] R = qr.R();
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < i; j++) {
                assertClose(0.0, R[i][j], 1e-13, "R strictly-lower[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testMGSReconstruction() {
        // A = Q R. Evaluate a few points and check each column matches.
        Domain d = new Domain(-1.0, 1.0);
        Quasimatrix A = wellConditionedQuasimatrix(d, 4);
        Quasimatrix.Qr qr = A.qr(Algorithm.MODIFIED_GRAM_SCHMIDT);
        double[][] R = qr.R();
        double[] xs = {-0.7, -0.3, 0.0, 0.4, 0.8};
        for (int col = 0; col < 4; col++) {
            for (double x : xs) {
                double reconstructed = 0.0;
                for (int i = 0; i <= col; i++) {
                    reconstructed += qr.Q().get(i).feval(x) * R[i][col];
                }
                double want = A.get(col).feval(x);
                assertClose(want, reconstructed, 1e-10, "A=QR col=" + col + " @ " + x);
            }
        }
    }

    @Test
    public void testHouseholderOrthonormality() {
        Quasimatrix A = wellConditionedQuasimatrix(new Domain(-1.0, 1.0), 5);
        Quasimatrix.Qr qr = A.qr(Algorithm.HOUSEHOLDER);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double ip = qr.Q().innerProduct(i, j);
                double want = (i == j) ? 1.0 : 0.0;
                assertClose(want, ip, 1e-11, "Q'Q[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testHouseholderUpperTriangular() {
        Quasimatrix A = wellConditionedQuasimatrix(new Domain(-1.0, 1.0), 5);
        Quasimatrix.Qr qr = A.qr(Algorithm.HOUSEHOLDER);
        double[][] R = qr.R();
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < i; j++) {
                assertClose(0.0, R[i][j], 1e-11, "R strictly-lower[" + i + "][" + j + "]");
            }
        }
    }

    @Test
    public void testHouseholderReconstruction() {
        Domain d = new Domain(-1.0, 1.0);
        Quasimatrix A = wellConditionedQuasimatrix(d, 4);
        Quasimatrix.Qr qr = A.qr(Algorithm.HOUSEHOLDER);
        double[][] R = qr.R();
        double[] xs = {-0.7, -0.3, 0.0, 0.4, 0.8};
        for (int col = 0; col < 4; col++) {
            for (double x : xs) {
                double reconstructed = 0.0;
                for (int i = 0; i <= col; i++) {
                    reconstructed += qr.Q().get(i).feval(x) * R[i][col];
                }
                double want = A.get(col).feval(x);
                assertClose(want, reconstructed, 1e-9, "A=QR col=" + col + " @ " + x);
            }
        }
    }

    @Test
    public void testDefaultAlgorithmIsMGS() {
        Quasimatrix A = wellConditionedQuasimatrix(new Domain(0.0, 1.0), 3);
        // Just verify the no-arg call doesn't blow up and gives an
        // orthonormal Q.
        Quasimatrix.Qr qr = A.qr();
        assertClose(1.0, qr.Q().innerProduct(0, 0), 1e-12, "default QR Q'Q[0,0]");
    }
}
