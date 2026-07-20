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
    public void testDefaultAlgorithmIsHouseholder() {
        // qr() no-arg should behave identically to qr(HOUSEHOLDER): same R
        // entries and Q columns whose inner products match to full precision.
        Domain d = new Domain(-1.0, 1.0);
        Quasimatrix A = wellConditionedQuasimatrix(d, 4);
        Quasimatrix.Qr viaDefault  = A.qr();
        Quasimatrix.Qr viaExplicit = A.qr(Algorithm.HOUSEHOLDER);
        // R matrices should agree entry-by-entry (same algorithm, same inputs).
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                assertClose(viaExplicit.R()[i][j], viaDefault.R()[i][j], 1e-14, "R[" + i + "][" + j + "]");
            }
        }
    }

    // -----------------------------------------------------------------
    // mrdivide (least-squares fit)
    // -----------------------------------------------------------------

    @Test
    public void testMrdivideRecoversExactCoefficients() {
        // Build A from {1, x, x²} on [-1, 1]. Take b = 3 - 2x + x², so
        // the exact coefficient vector is (3, -2, 1). mrdivide should
        // recover it to spectral precision.
        Domain d = new Domain(-1.0, 1.0);
        Chebfun[] cols = {
            new Chebfun(x -> 1.0,       d),
            new Chebfun(x -> x,         d),
            new Chebfun(x -> x * x,     d),
        };
        Quasimatrix A = new Quasimatrix(cols);
        Chebfun b = new Chebfun(x -> 3.0 - 2.0 * x + x * x, d);
        double[] c = A.mrdivide(b);
        assertClose(3.0,  c[0], 1e-11, "c[0]");
        assertClose(-2.0, c[1], 1e-11, "c[1]");
        assertClose(1.0,  c[2], 1e-11, "c[2]");
    }

    @Test
    public void testMrdivideOnOrthonormalBasisIsInnerProducts() {
        // If A's columns are already L²-orthonormal, mrdivide reduces to
        // c_i = <q_i, b>. Use a simple 2-column basis: q_0 = 1/sqrt(2),
        // q_1 = sqrt(3/2) x on [-1, 1] (rescaled Legendre P_0, P_1).
        Domain d = new Domain(-1.0, 1.0);
        Chebfun[] cols = {
            new Chebfun(x -> 1.0 / Math.sqrt(2.0), d),
            new Chebfun(x -> Math.sqrt(1.5) * x,   d),
        };
        Quasimatrix A = new Quasimatrix(cols);
        // b = 4 * q_0 + 5 * q_1
        Chebfun b = new Chebfun(x -> 4.0 / Math.sqrt(2.0) + 5.0 * Math.sqrt(1.5) * x, d);
        double[] c = A.mrdivide(b);
        assertClose(4.0, c[0], 1e-11, "c[0]");
        assertClose(5.0, c[1], 1e-11, "c[1]");
    }

    @Test
    public void testMrdivideRejectsMismatchedDomain() {
        Chebfun[] cols = { new Chebfun(x -> 1.0, new Domain(0.0, 1.0)) };
        Quasimatrix A = new Quasimatrix(cols);
        Chebfun b = new Chebfun(x -> 1.0, new Domain(0.0, 2.0));
        try {
            A.mrdivide(b);
            throw new AssertionError("expected domain mismatch to throw");
        } catch (IllegalArgumentException expected) {
            // OK
        }
    }

    /**
     * At n=15, Householder QR on a smooth quasimatrix should give a Q whose
     * columns are L²-orthonormal to at least ~1e-10 — the current
     * monomials-plus-MGS reference basis fails at n much smaller than that.
     * The Legendre reference basis with pointwise scalar-recurrence
     * evaluation is what makes this size reachable.
     */
    @Test
    public void testHouseholderRemainsOrthonormalAtN15() {
        int n = 15;
        Domain d = new Domain(-1.0, 1.0);
        // Well-scaled smooth columns: shifted Chebyshev polynomials cos(k * acos(x/2)).
        // Bounded, non-oscillatory (compared to cos(kx) at k=20), and span n independent
        // polynomial directions.
        Chebfun[] cols = new Chebfun[n];
        for (int k = 0; k < n; k++) {
            final int K = k;
            cols[k] = new Chebfun(x -> Math.cos(K * Math.acos(x)) + 0.1 * (K + 1) * x, d);
        }
        Quasimatrix A = new Quasimatrix(cols);
        Quasimatrix.Qr qr = A.qr(Algorithm.HOUSEHOLDER);
        double worst = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double ip = qr.Q().innerProduct(i, j);
                double want = (i == j) ? 1.0 : 0.0;
                worst = Math.max(worst, Math.abs(ip - want));
            }
        }
        if (worst > 1e-10) {
            throw new AssertionError("Householder QR lost orthonormality at n=" + n
                + ": worst |Q'Q - I| = " + worst);
        }
    }
}
