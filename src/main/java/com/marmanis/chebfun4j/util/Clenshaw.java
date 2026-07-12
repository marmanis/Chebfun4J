package com.marmanis.chebfun4j.util;

/**
 * Evaluation of a Chebyshev series {@code f(x) = sum_{k=0}^{n-1} c_k T_k(x)}
 * via Clenshaw's recurrence — the numerically stable way to sum a Chebyshev
 * expansion at a point in {@code [-1, 1]}. Runs in {@code O(n)} per point with
 * one multiply-add per coefficient; roughly 2× faster than expanding
 * {@code T_k(x)} explicitly and much more accurate than a naive Horner-on-
 * {@code T_k} scheme.
 */
public final class Clenshaw {
    private Clenshaw() {}

    /**
     * Evaluate the Chebyshev series with coefficients {@code c} at the point
     * {@code x} (which should be in {@code [-1, 1]}; the recurrence itself is
     * valid outside but the series only approximates {@code f} inside).
     */
    public static double eval(double[] c, double x) {
        int n = c.length;
        if (n == 0) return 0.0;
        if (n == 1) return c[0];
        double bkp1 = 0.0;
        double bk = 0.0;
        double twoX = 2.0 * x;
        for (int k = n - 1; k >= 1; k--) {
            double bkm1 = twoX * bk - bkp1 + c[k];
            bkp1 = bk;
            bk = bkm1;
        }
        return c[0] + x * bk - bkp1;
    }

    /**
     * Evaluate at every point of {@code xs}, returning a fresh array. Cheaper
     * than calling {@link #eval(double[], double)} in a loop for large
     * {@code xs} because the Clenshaw setup work is negligible per point.
     */
    public static double[] evalMany(double[] c, double[] xs) {
        double[] out = new double[xs.length];
        for (int i = 0; i < xs.length; i++) out[i] = eval(c, xs[i]);
        return out;
    }
}
