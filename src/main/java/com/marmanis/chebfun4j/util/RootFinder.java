package com.marmanis.chebfun4j.util;

import com.marmanis.chebfun4j.Chebtech;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Real rootfinding for a {@link Chebtech} on the reference interval
 * {@code [-1, 1]} via the <em>colleague matrix</em>: the eigenvalues of an
 * {@code n×n} matrix directly built from the Chebyshev coefficients give
 * exactly the roots of the corresponding degree-{@code n} polynomial
 * (Boyd 2014, Trefethen 2013). Real eigenvalues in {@code [-1, 1]} are
 * kept; complex ones and real ones outside the interval are discarded.
 *
 * <p>This finds all roots regardless of multiplicity — including even-
 * multiplicity roots like {@code x^2}'s zero at 0 that the old sign-change
 * bracketing missed — because the eigenvalue formulation is oblivious to
 * sign changes.
 *
 * <p>For high-degree Chebtechs (length {@code > 100}) the {@code O(n^3)}
 * eigenvalue solve gets expensive and can accumulate roundoff, so we
 * recursively subdivide the reference interval, restrict the polynomial
 * to each half by Chebyshev interpolation on the halved grid, and recurse
 * — the standard chebfun approach.
 */
public final class RootFinder {
    private RootFinder() {}

    /** Above this length we split the interval and recurse. */
    private static final int SPLIT_THRESHOLD = 100;

    /** Simplification tolerance before rootfinding — drops noise-level tail. */
    private static final double SIMPLIFY_TOL = 1e-13;

    /**
     * How much slack outside {@code [-1, 1]} to accept as a real interior
     * root. Colleague eigenvalues can drift a few ulp past the endpoint even
     * for exact-endpoint roots (like {@code sin(10 pi x)}'s zeros at ±1),
     * so we accept up to {@code 1e-6} and clamp back to {@code [-1, 1]}.
     */
    private static final double EDGE_TOL = 1e-6;

    /**
     * Any eigenvalue with {@code |imag| > this * (1 + |real|)} is discarded.
     * A repeated real eigenvalue (e.g. the double root of {@code (x-r)^2})
     * comes out with imaginary parts on the order of {@code sqrt(eps) ~=
     * 1.5e-8}, so we loosen to {@code 1e-6} — this is still tight enough
     * to reject genuine complex roots, whose imaginary parts are typically
     * order {@code 1} for a normalised polynomial.
     */
    private static final double IMAG_TOL = 1e-6;

    /**
     * Roots of {@code t} in {@code [-1, 1]}, sorted ascending. Even-
     * multiplicity roots are returned once (the eigenvalue solver returns
     * the algebraic multiplicity, but chebfun's contract is to report each
     * distinct root once — matching MATLAB chebfun).
     */
    public static double[] rootsOnRef(Chebtech t) {
        Chebtech simplified = t.simplify(SIMPLIFY_TOL);
        int n = simplified.length();
        if (n <= 1) {
            return new double[0]; // constant — no discrete roots
        }
        if (n <= SPLIT_THRESHOLD) {
            return colleagueRoots(simplified);
        }
        return subdividedRoots(simplified);
    }

    private static double[] colleagueRoots(Chebtech t) {
        double[] c = t.coeffs();
        int n = c.length - 1; // matrix size is degree = length - 1
        if (n <= 0) return new double[0];
        double cN = c[n];
        // Colleague matrix M is n×n. Multiplication by x on the Chebyshev
        // basis, on the quotient ring modulo p(x). See class Javadoc.
        double[] a = new double[n * n];
        // Column 0: M[1][0] = 1.
        if (n >= 2) a[1 * n + 0] = 1.0;
        else a[0] = -c[0] / cN; // n == 1: single-variable case, root = -c[0]/c[1]
        // Columns 1..n-2: tridiagonal 1/2 above and below diagonal.
        for (int k = 1; k <= n - 2; k++) {
            a[(k - 1) * n + k] = 0.5;
            a[(k + 1) * n + k] = 0.5;
        }
        // Column n-1: -c[i]/(2 c_n) everywhere, plus 1/2 in row n-2.
        if (n >= 2) {
            double denom = 2.0 * cN;
            for (int i = 0; i < n; i++) {
                a[i * n + (n - 1)] = -c[i] / denom;
            }
            a[(n - 2) * n + (n - 1)] += 0.5;
        }
        NDArray A = new ConcreteNDArray(a, new Shape(n, n));
        NDArray[] w = Linalg.eig(A);
        double[] wr = w[0].toDoubleArray();
        double[] wi = w[1].toDoubleArray();
        List<Double> found = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            if (Math.abs(wi[k]) > IMAG_TOL * (1.0 + Math.abs(wr[k]))) continue;
            double x = wr[k];
            if (x < -1.0 - EDGE_TOL || x > 1.0 + EDGE_TOL) continue;
            if (x < -1.0) x = -1.0;
            if (x > 1.0) x = 1.0;
            found.add(x);
        }
        return sortedDedup(found);
    }

    private static double[] subdividedRoots(Chebtech t) {
        // Halve the sample length on each level so the recursion terminates
        // even if simplify() can't shrink the coefficient tail (which
        // happens for pieces that were fit through a kink and never really
        // resolved — see Chebfun's splitting-on constructor). Restricting
        // to a sub-interval that is smoother than the full interval is
        // exactly when a shorter polynomial approximation is adequate.
        int n = Math.max((t.length() + 1) / 2, SPLIT_THRESHOLD + 1);
        n = ChebyshevPoints.nextValidLength(n);
        Chebtech left = restrict(t, -1.0, 0.0, n);
        Chebtech right = restrict(t, 0.0, 1.0, n);
        double[] leftRoots = rootsOnRef(left);
        double[] rightRoots = rootsOnRef(right);
        List<Double> merged = new ArrayList<>(leftRoots.length + rightRoots.length);
        for (double y : leftRoots) merged.add(0.5 * (y - 1.0));  // map back to [-1, 0]
        for (double y : rightRoots) merged.add(0.5 * (y + 1.0)); // map back to [0, 1]
        return sortedDedup(merged);
    }

    /**
     * Return a length-{@code n} Chebtech representing {@code t} restricted
     * to the sub-interval {@code [a, b] subset [-1, 1]}, expressed on its
     * own local reference {@code [-1, 1]}. Sampling drives the length
     * (rather than inheriting from {@code t}) so subdivision recursion
     * strictly reduces work per level.
     */
    private static Chebtech restrict(Chebtech t, double a, double b, int n) {
        double[] refGrid = ChebyshevPoints.secondKind(n);
        double[] vals = new double[n];
        double mid = 0.5 * (a + b);
        double halfWidth = 0.5 * (b - a);
        for (int i = 0; i < n; i++) {
            double x = mid + halfWidth * refGrid[i];
            vals[i] = t.eval(x);
        }
        return Chebtech.fromValues(vals);
    }

    private static double[] sortedDedup(List<Double> roots) {
        Collections.sort(roots);
        double eps = 1e-11;
        List<Double> dedup = new ArrayList<>(roots.size());
        double last = Double.NEGATIVE_INFINITY;
        for (double r : roots) {
            if (r - last > eps) {
                dedup.add(r);
                last = r;
            }
        }
        double[] out = new double[dedup.size()];
        for (int i = 0; i < out.length; i++) out[i] = dedup.get(i);
        return out;
    }
}
