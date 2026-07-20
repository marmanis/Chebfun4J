package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ArrayMath;
import com.marmanis.chebfun4j.util.DifferentiationMatrix;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * Initial-value problem solver:
 * <pre>
 *   u'(x) = f(x, u(x)),   x ∈ [a, b],
 *   u(a) = u₀.
 * </pre>
 * Chebyshev spectral collocation — the same discretization
 * {@link Chebop} uses for BVPs, adapted for a first-order equation with a
 * single initial condition. On a length-{@code n+1} second-kind grid,
 * {@code u} is represented by its grid values and {@code u'} by the scaled
 * spectral differentiation matrix {@code D} times those values. The initial
 * condition replaces one row of the residual system, pinning
 * {@code u(a) = u₀}; the remaining {@code n} rows enforce
 * {@code (D u)_i - f(x_i, u_i) = 0}.
 *
 * <p>For <b>linear</b> IVPs — {@code f(x, u) = a(x) u + b(x)} — this is a
 * single linear solve per grid size. For <b>nonlinear</b> IVPs the solver
 * runs Newton iterations, using finite-difference partials
 * {@code ∂f/∂u} by default (users can override with analytic partials via
 * {@link Residual}). Grid sizes {@code 16, 32, 64, ..., 512} are tried in
 * turn until successive iterates agree to {@code tol}.
 *
 * <p>Row 0 of the Chebyshev-2nd-kind grid corresponds to {@code x = b}
 * (the right endpoint) and row {@code n} to {@code x = a} (the left) —
 * this is the standard chebfun ordering. The IC row therefore goes at
 * index {@code n}, not 0.
 */
public final class Ivp {
    /** Max grid the adaptive loop tries. */
    public static final int MAX_N = 512;
    public static final double DEFAULT_TOL = 1e-10;
    private static final int[] GRID_SIZES = {16, 32, 64, 128, 256, MAX_N};
    private static final int MAX_NEWTON_ITER = 50;
    private static final double NEWTON_TOL = 1e-12;

    /**
     * Residual view: value {@code f(x, u)} plus the partial derivative
     * {@code ∂f/∂u}. Users can override the default finite-difference
     * partial for analytic accuracy or when {@code f} is not smooth.
     */
    public interface Residual {
        double f(double x, double u);

        /** Default: centered FD around {@code u}. */
        default double dU(double x, double u) {
            double h = Math.cbrt(Math.ulp(1.0)) * Math.max(1.0, Math.abs(u));
            return (f(x, u + h) - f(x, u - h)) / (2 * h);
        }
    }

    private Ivp() {}

    /**
     * Solve {@code u' = F.f(x, u)} on {@code domain} with {@code u(a) = u0}.
     * Adaptive over grid size; returns as soon as successive grids agree to
     * {@link #DEFAULT_TOL}. If none converge by {@link #MAX_N}, returns the
     * {@link #MAX_N} answer. A pure lambda {@code (x, u) -> ...} resolves
     * to {@link Residual} (the finite-difference default {@code ∂f/∂u}
     * applies); pass a full {@link Residual} instance to supply analytic
     * partials.
     */
    public static Chebfun solve(Residual F, Domain domain, double u0) {
        return solve(F, domain, u0, DEFAULT_TOL);
    }

    public static Chebfun solve(Residual F, Domain domain, double u0, double tol) {
        Chebfun last = null;
        for (int n : GRID_SIZES) {
            double[] uVals = solveAt(F, domain, u0, n, last);
            Chebfun candidate = new Chebfun(Chebtech.fromValues(uVals), domain);
            if (last != null) {
                double vscale = Math.max(1.0, candidate.normInf());
                double err = candidate.minus(last).normInf();
                if (err <= tol * vscale) return candidate;
            }
            last = candidate;
        }
        return last;
    }

    /**
     * Fixed-grid solve. Newton iteration on the collocation system, seeded
     * by the previous iterate interpolated (via {@code feval}) onto this
     * grid if available, else by the constant {@code u0}.
     */
    private static double[] solveAt(Residual F, Domain domain, double u0, int n, Chebfun seed) {
        int size = n + 1;
        double[] scaledD = scaledDMatrix(n, domain);
        double[] xGrid = physicalGrid(n, domain);

        double[] u = new double[size];
        if (seed != null) {
            for (int i = 0; i < size; i++) u[i] = seed.feval(xGrid[i]);
        } else {
            java.util.Arrays.fill(u, u0);
        }
        // Force the initial condition on the seed too, so Newton starts
        // from a feasible boundary value.
        u[n] = u0;

        double[] r = new double[size];
        double[] J = new double[size * size];
        for (int iter = 0; iter < MAX_NEWTON_ITER; iter++) {
            // Residual: (D u)_i - f(x_i, u_i), IC row at index n replaced
            // by u[n] - u0.
            double[] Du = matVec(scaledD, u, size);
            for (int i = 0; i < size; i++) r[i] = Du[i] - F.f(xGrid[i], u[i]);
            r[n] = u[n] - u0;

            double resNorm = ArrayMath.maxAbs(r);
            if (resNorm < NEWTON_TOL) return u;

            // Jacobian J[i][j] = D[i][j] - δ_ij * ∂f/∂u(x_i, u_i).
            for (int i = 0; i < size * size; i++) J[i] = scaledD[i];
            for (int i = 0; i < size; i++) {
                J[i * size + i] -= F.dU(xGrid[i], u[i]);
            }
            // IC row overrides.
            for (int j = 0; j < size; j++) J[n * size + j] = 0.0;
            J[n * size + n] = 1.0;

            double[] negR = new double[size];
            for (int i = 0; i < size; i++) negR[i] = -r[i];
            double[] du;
            try {
                NDArray Jarr = new ConcreteNDArray(J, new Shape(size, size));
                NDArray rArr = new ConcreteNDArray(negR, new Shape(size));
                du = Linalg.solve(Jarr, rArr).toDoubleArray();
            } catch (RuntimeException e) {
                return u;
            }
            for (int i = 0; i < size; i++) u[i] += du[i];

            double duNorm = ArrayMath.maxAbs(du);
            double uScale = Math.max(1.0, ArrayMath.maxAbs(u));
            if (duNorm <= NEWTON_TOL * uScale) return u;
        }
        return u;
    }

    private static double[] scaledDMatrix(int n, Domain domain) {
        int size = n + 1;
        double[] D = DifferentiationMatrix.chebD(n);
        double jac = 2.0 / domain.length();
        double[] out = new double[size * size];
        for (int i = 0; i < size * size; i++) out[i] = jac * D[i];
        return out;
    }

    private static double[] physicalGrid(int n, Domain domain) {
        double[] out = new double[n + 1];
        for (int j = 0; j <= n; j++) out[j] = domain.fromRef(Math.cos(Math.PI * j / n));
        return out;
    }

    private static double[] matVec(double[] mat, double[] v, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) s += mat[i * n + j] * v[j];
            out[i] = s;
        }
        return out;
    }
}
