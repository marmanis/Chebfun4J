package com.marmanis.chebfun4j.util;

/**
 * Chebyshev nodes ("grid points") used by {@code chebtech2} / chebfun4j.
 *
 * <p>The <em>second-kind</em> points are the extrema of the Chebyshev
 * polynomial {@code T_{N-1}}, plus the endpoints, i.e.
 * {@code x_k = cos(pi * k / (N - 1))} for {@code k = 0 .. N-1}, so
 * {@code x_0 = 1} and {@code x_{N-1} = -1}. They are the natural grid for the
 * discrete cosine transform of type I (DCT-I) that maps values ↔ Chebyshev
 * coefficients — see {@link ChebTransform}. Chebfun's adaptive constructor
 * doubles the grid length through {@code 1, 3, 5, 9, 17, 33, ...}
 * ({@code 2^k + 1}) to keep the DCT-I FFT-based.
 */
public final class ChebyshevPoints {
    private ChebyshevPoints() {}

    /**
     * Chebyshev-2nd-kind points of length {@code n} on {@code [-1, 1]},
     * ordered {@code x_0 = 1, x_1 = cos(pi/(n-1)), ..., x_{n-1} = -1}. For
     * {@code n == 1} returns {@code {0}} (the natural single-point "grid" at
     * the interval midpoint).
     */
    public static double[] secondKind(int n) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1, got " + n);
        if (n == 1) return new double[]{0.0};
        double[] x = new double[n];
        double denom = n - 1;
        for (int k = 0; k < n; k++) {
            x[k] = Math.cos(Math.PI * k / denom);
        }
        // Nail down the exact endpoints; the cos formula gives ±1 up to
        // rounding at k = 0 and k = n - 1 already, but chebfun users rely on
        // exact endpoint values for feval at ±1.
        x[0] = 1.0;
        x[n - 1] = -1.0;
        return x;
    }

    /**
     * Smallest length {@code L >= n} such that {@code L == 1} or
     * {@code L - 1} is a power of two — i.e. the smallest {@code 2^k + 1}
     * grid that admits an FFT-based DCT-I. Used to round the length of a
     * newly-constructed or resized Chebtech up to the fast path.
     */
    public static int nextValidLength(int n) {
        if (n <= 1) return 1;
        int m = n - 1;
        int p = Integer.highestOneBit(m);
        return (p == m) ? n : (p << 1) + 1;
    }
}
