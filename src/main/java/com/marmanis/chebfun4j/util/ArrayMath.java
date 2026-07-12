package com.marmanis.chebfun4j.util;

import com.marmanis.chebfun4j.Chebfun;

/**
 * Small numerical helpers shared by the adaptive solvers ({@code Chebop},
 * {@code NonlinearChebop}, {@code LinearBlockChebop}, {@code NonlinearSystem}).
 * All operate on plain {@code double[]} arrays or on {@link Chebfun}s via
 * {@code feval}; nothing here reaches into {@link Chebfun}'s internals.
 */
public final class ArrayMath {
    private ArrayMath() {}

    /** Maximum absolute value in {@code v}. Returns {@code 0} for an empty array. */
    public static double maxAbs(double[] v) {
        double m = 0.0;
        for (double x : v) {
            double ax = Math.abs(x);
            if (ax > m) m = ax;
        }
        return m;
    }

    /**
     * {@code max_{i < k} |a[i] - b[i]|}, clamped to the actual lengths of
     * {@code a} and {@code b}. Used by the adaptive-grid eigenvalue loop
     * to compare successive eigenvalue lists (the first {@code k} entries
     * only).
     */
    public static double maxAbsDiff(double[] a, double[] b, int k) {
        int n = Math.min(a.length, Math.min(b.length, k));
        double err = 0.0;
        for (int i = 0; i < n; i++) {
            double d = Math.abs(a[i] - b[i]);
            if (d > err) err = d;
        }
        return err;
    }

    /**
     * Approximate uniform-grid probe of {@code max |a(x) - b(x)|} over the
     * shared domain. Samples at {@code 4 × min(a.length, b.length)} equally-
     * spaced points (with a floor of 32), evaluating both Chebfuns via
     * {@code feval} at each. This is a sampled residual — good enough for
     * a convergence check between adaptive-grid iterates, but not exact;
     * {@code a.minus(b).normInf()} would be exact at higher cost.
     */
    public static double maxAbsDiff(Chebfun a, Chebfun b) {
        int n = Math.min(a.length(), b.length()) * 4;
        n = Math.max(n, 32);
        double lo = a.domain().a();
        double hi = a.domain().b();
        double dx = (hi - lo) / (n - 1);
        double max = 0.0;
        for (int i = 0; i < n; i++) {
            double x = lo + i * dx;
            double d = Math.abs(a.feval(x) - b.feval(x));
            if (d > max) max = d;
        }
        return max;
    }
}
