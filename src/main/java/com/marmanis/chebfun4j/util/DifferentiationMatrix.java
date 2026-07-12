package com.marmanis.chebfun4j.util;

/**
 * Chebyshev spectral differentiation matrix at the second-kind grid
 * {@code x_j = cos(pi j / n)} for {@code j = 0..n} on the reference
 * interval {@code [-1, 1]} — the classical construction from Trefethen,
 * <em>Spectral Methods in MATLAB</em> (2000), routine {@code cheb.m}.
 *
 * <p>Given values {@code v_j = f(x_j)} at those points, the matrix-vector
 * product {@code D * v} gives approximate values of {@code f'} at the same
 * points, exact for any polynomial of degree {@code &lt;= n}.
 */
public final class DifferentiationMatrix {
    private DifferentiationMatrix() {}

    /**
     * The {@code (n+1) x (n+1)} differentiation matrix on {@code [-1, 1]}
     * at the Chebyshev-2nd-kind grid of length {@code n + 1}, returned in
     * row-major layout.
     */
    public static double[] chebD(int n) {
        int size = n + 1;
        double[] D = new double[size * size];
        double[] x = new double[size];
        double[] c = new double[size];
        for (int j = 0; j <= n; j++) x[j] = Math.cos(Math.PI * j / n);
        for (int j = 0; j <= n; j++) c[j] = (j == 0 || j == n) ? 2.0 : 1.0;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n; j++) {
                if (i != j) {
                    double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                    D[i * size + j] = c[i] * sign / (c[j] * (x[i] - x[j]));
                }
            }
        }
        // Diagonal: fill via negative-row-sum trick to preserve exact
        // differentiation of constants (row-sum should be zero).
        for (int i = 0; i <= n; i++) {
            double rowSum = 0.0;
            for (int j = 0; j <= n; j++) if (i != j) rowSum += D[i * size + j];
            D[i * size + i] = -rowSum;
        }
        return D;
    }

    /**
     * Matrix square: {@code D^2}, computed as a straightforward {@code O(n^3)}
     * multiplication. For higher powers, iterate.
     */
    public static double[] matMul(double[] a, double[] b, int n) {
        double[] c = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double aik = a[i * n + k];
                if (aik == 0.0) continue;
                for (int j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j];
            }
        }
        return c;
    }
}
