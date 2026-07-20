package com.marmanis.chebfun4j.util;

import java.util.function.DoubleUnaryOperator;

/**
 * Natural-boundary cubic-spline interpolation of a set of sample points
 * {@code (x_i, y_i)}, {@code i = 0..n-1}, with strictly ascending
 * {@code x_i}. A single class-level {@link DoubleUnaryOperator} view is
 * exposed via {@link #asFunction()}; that view returns the interpolant
 * inside {@code [x_0, x_{n-1}]} and clamps to the boundary values outside.
 *
 * <p>Small, self-contained implementation using the classic tridiagonal
 * solve for the second-derivative moments (see Press et al., <em>Numerical
 * Recipes</em> §3.3). Natural boundary conditions ({@code S''(x_0) =
 * S''(x_{n-1}) = 0}) — the simplest choice, and adequate for feeding a
 * downstream {@link com.marmanis.chebfun4j.Chebfun} adaptive constructor
 * that will resolve the (piecewise cubic) interpolant to spectral
 * precision on each smooth region.
 */
public final class CubicSpline {
    private final double[] x;
    private final double[] y;
    private final double[] m; // second derivatives at knots

    private CubicSpline(double[] x, double[] y, double[] m) {
        this.x = x;
        this.y = y;
        this.m = m;
    }

    /**
     * Build a natural cubic spline through {@code (x[i], y[i])}. Requires
     * at least two points, strictly ascending {@code x}, and matching
     * array lengths.
     */
    public static CubicSpline of(double[] x, double[] y) {
        if (x.length < 2) throw new IllegalArgumentException("need at least 2 points, got " + x.length);
        if (x.length != y.length) {
            throw new IllegalArgumentException(
                "x and y must have matching lengths, got " + x.length + " vs " + y.length);
        }
        for (int i = 1; i < x.length; i++) {
            if (!(x[i] > x[i - 1])) {
                throw new IllegalArgumentException(
                    "x must be strictly ascending; violated at index " + i);
            }
        }
        int n = x.length;
        double[] xs = x.clone();
        double[] ys = y.clone();
        double[] m = new double[n];
        if (n == 2) {
            // Linear segment — both moments are zero under the natural BC.
            return new CubicSpline(xs, ys, m);
        }
        // Thomas algorithm on the tridiagonal moment system, sub/super
        // diagonals of size n-2 (interior nodes only).
        double[] sub = new double[n];
        double[] diag = new double[n];
        double[] sup = new double[n];
        double[] rhs = new double[n];
        // Natural BCs pin m[0] = m[n-1] = 0.
        diag[0] = 1.0;
        rhs[0] = 0.0;
        diag[n - 1] = 1.0;
        rhs[n - 1] = 0.0;
        for (int i = 1; i < n - 1; i++) {
            double hL = xs[i] - xs[i - 1];
            double hR = xs[i + 1] - xs[i];
            sub[i] = hL;
            diag[i] = 2.0 * (hL + hR);
            sup[i] = hR;
            rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / hR - (ys[i] - ys[i - 1]) / hL);
        }
        // Forward sweep.
        for (int i = 1; i < n; i++) {
            double factor = sub[i] / diag[i - 1];
            diag[i] -= factor * sup[i - 1];
            rhs[i]  -= factor * rhs[i - 1];
        }
        // Back-substitute.
        m[n - 1] = rhs[n - 1] / diag[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            m[i] = (rhs[i] - sup[i] * m[i + 1]) / diag[i];
        }
        return new CubicSpline(xs, ys, m);
    }

    /** Leftmost knot. */
    public double xMin() { return x[0]; }
    /** Rightmost knot. */
    public double xMax() { return x[x.length - 1]; }

    /**
     * Evaluate the spline at {@code xq}. Outside {@code [xMin, xMax]}
     * clamps to the boundary values (returns {@code y[0]} or
     * {@code y[n-1]}) rather than extrapolating cubically.
     */
    public double eval(double xq) {
        int n = x.length;
        if (xq <= x[0]) return y[0];
        if (xq >= x[n - 1]) return y[n - 1];
        // Binary-search for the containing segment.
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (x[mid] > xq) hi = mid;
            else lo = mid;
        }
        double h = x[hi] - x[lo];
        double a = (x[hi] - xq) / h;
        double b = (xq - x[lo]) / h;
        return a * y[lo] + b * y[hi]
            + ((a * a * a - a) * m[lo] + (b * b * b - b) * m[hi]) * (h * h) / 6.0;
    }

    /** {@link DoubleUnaryOperator} view of {@link #eval(double)}. */
    public DoubleUnaryOperator asFunction() {
        return this::eval;
    }
}
