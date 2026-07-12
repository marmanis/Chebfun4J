package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ArrayMath;
import com.marmanis.chebfun4j.util.DifferentiationMatrix;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * A multi-component nonlinear ODE boundary-value problem
 * <pre>
 *   F(x, u(x), u'(x), u''(x)) = 0,
 *   u(x) = [u_0, u_1, ..., u_{k-1}]  a vector of unknowns.
 * </pre>
 * The residual {@code F} is pointwise ({@link Residual}): given
 * {@code (x, u, u', u'')}, it returns a {@code double[k]} residual
 * vector. Newton iteration on a Chebyshev-spectral-collocation grid
 * solves the linearized block system
 * {@code J Δu = -r}, where the block Jacobian at grid row {@code p},
 * block row {@code i}, block column {@code j} is
 * <pre>
 *   J[i, j][p, q] = ∂F_i/∂u_j     * δ_{pq}
 *                 + ∂F_i/∂u'_j    * D[p, q]
 *                 + ∂F_i/∂u''_j   * D²[p, q].
 * </pre>
 * The three per-component partials are computed by centered finite
 * differences on {@link Residual#at} by default; users can override the
 * default {@link Residual#jacobian} for analytic Jacobians.
 *
 * <p>Uses the same linear {@link SystemBC} type as
 * {@link LinearBlockChebop}. Adaptive-{@code n} loop over
 * {@code 16, 32, ..., 512}; each new grid seeds Newton with the
 * previous grid's solution.
 */
public final class NonlinearSystem {
    public static final int MAX_N = 512;
    public static final double DEFAULT_TOL = 1e-10;
    private static final int[] GRID_SIZES = {16, 32, 64, 128, 256, MAX_N};
    private static final int MAX_BACKTRACKS = 5;

    /**
     * Pointwise residual for a {@code k}-component system:
     * {@code F(x, u, u', u'') = 0} where {@code u}, {@code u'},
     * {@code u''} are length-{@code k} arrays and the return value is a
     * length-{@code k} residual vector.
     */
    public interface Residual {
        int numComponents();

        /** {@code F(x, u, u', u'')} as a length-{@code k} vector. */
        double[] at(double x, double[] u, double[] up, double[] upp);

        /**
         * The three {@code k × k} Jacobian blocks
         * {@code {∂F/∂u, ∂F/∂u', ∂F/∂u''}} at the given point.
         * Default: centered finite differences on {@link #at}. Override
         * for analytic accuracy.
         */
        default double[][][] jacobian(double x, double[] u, double[] up, double[] upp) {
            int k = numComponents();
            double[][] Ju   = new double[k][k];
            double[][] Jup  = new double[k][k];
            double[][] Jupp = new double[k][k];
            for (int j = 0; j < k; j++) {
                double[] uP  = u.clone(), uM  = u.clone();
                double hU = fdStep(u[j]);   uP[j]  += hU;  uM[j]  -= hU;
                double[] fPu = at(x, uP, up, upp), fMu = at(x, uM, up, upp);
                double[] upP = up.clone(), upM = up.clone();
                double hUp = fdStep(up[j]); upP[j] += hUp; upM[j] -= hUp;
                double[] fPup = at(x, u, upP, upp), fMup = at(x, u, upM, upp);
                double[] uppP = upp.clone(), uppM = upp.clone();
                double hUpp = fdStep(upp[j]); uppP[j] += hUpp; uppM[j] -= hUpp;
                double[] fPupp = at(x, u, up, uppP), fMupp = at(x, u, up, uppM);
                for (int i = 0; i < k; i++) {
                    Ju  [i][j] = (fPu [i] - fMu [i]) / (2 * hU);
                    Jup [i][j] = (fPup[i] - fMup[i]) / (2 * hUp);
                    Jupp[i][j] = (fPupp[i] - fMupp[i]) / (2 * hUpp);
                }
            }
            return new double[][][]{Ju, Jup, Jupp};
        }

        private static double fdStep(double x) {
            return Math.cbrt(Math.ulp(1.0)) * Math.max(1.0, Math.abs(x));
        }
    }

    private final Domain domain;
    private final Residual residual;

    public NonlinearSystem(Domain domain, Residual residual) {
        this.domain = domain;
        this.residual = residual;
    }

    public Domain domain() { return domain; }
    public int numComponents() { return residual.numComponents(); }

    /** Solve with default Newton options. */
    public Chebfun[] solve(SystemBC[] bcA, SystemBC[] bcB) {
        return solve(bcA, bcB, NewtonOptions.defaults());
    }

    public Chebfun[] solve(SystemBC[] bcA, SystemBC[] bcB, NewtonOptions opts) {
        int k = numComponents();
        Chebfun[] last = null;
        for (int n : GRID_SIZES) {
            double[][] uVals = solveAt(n, bcA, bcB, opts, last);
            Chebfun[] candidate = new Chebfun[k];
            for (int c = 0; c < k; c++) {
                candidate[c] = new Chebfun(Chebtech.fromValues(uVals[c]), domain);
            }
            if (last != null) {
                double err = 0.0, scale = 0.0;
                for (int c = 0; c < k; c++) {
                    err = Math.max(err, ArrayMath.maxAbsDiff(candidate[c], last[c]));
                    scale = Math.max(scale, candidate[c].normInf());
                }
                if (err <= opts.tol() * Math.max(1.0, scale)) return candidate;
            }
            last = candidate;
        }
        return last;
    }

    /**
     * Newton loop at a fixed grid size {@code n}. Returns
     * {@code uVals[c][p]} — the value of component {@code c} at grid
     * point {@code p}. The {@code previousSolution} array, when
     * non-null, seeds Newton via evaluation on the new grid; otherwise
     * we start from zero.
     */
    private double[][] solveAt(int n, SystemBC[] bcA, SystemBC[] bcB,
                               NewtonOptions opts, Chebfun[] previousSolution) {
        int size = n + 1;
        int k = numComponents();
        int total = k * size;

        double[] scaledD = scaledDMatrix(n);
        double[] scaledD2 = DifferentiationMatrix.matMul(scaledD, scaledD, size);
        double[] xGrid = physicalGrid(n);

        // Initial guess. Row-major: uFlat[c * size + p].
        double[] uFlat = seedInitialGuess(size, k, xGrid, opts, previousSolution);

        // Precompute BC row indices and BC-row linear forms once.
        BcRow[] bcRows = buildBcRows(bcA, bcB, scaledD, size, total, n, k);

        double damping = opts.initialDamping();
        for (int iter = 0; iter < opts.maxIter(); iter++) {
            // u(x), u'(x), u''(x) at grid points — a k-vector per point.
            double[][] u   = componentGrids(uFlat, k, size);
            double[][] up  = componentDerivatives(scaledD,  u, k, size);
            double[][] upp = componentDerivatives(scaledD2, u, k, size);

            double[] r = new double[total];
            for (int p = 0; p < size; p++) {
                double[] uP   = column(u,   p, k);
                double[] upP  = column(up,  p, k);
                double[] uppP = column(upp, p, k);
                double[] fp = residual.at(xGrid[p], uP, upP, uppP);
                for (int i = 0; i < k; i++) r[i * size + p] = fp[i];
            }
            // BC residuals overwrite specific rows.
            for (BcRow row : bcRows) {
                r[row.globalRow] = row.dot(uFlat) - row.value;
            }

            double resNorm = ArrayMath.maxAbs(r);
            if (resNorm < opts.tol()) return extractComponents(uFlat, k, size);

            // Block Jacobian assembly. For each grid point p, compute the
            // three k×k Jacobian blocks (∂F/∂u, ∂F/∂u', ∂F/∂u'') and
            // scatter into the big matrix.
            double[] J = new double[total * total];
            for (int p = 0; p < size; p++) {
                double[] uP   = column(u,   p, k);
                double[] upP  = column(up,  p, k);
                double[] uppP = column(upp, p, k);
                double[][][] jac = residual.jacobian(xGrid[p], uP, upP, uppP);
                double[][] Ju = jac[0], Jup = jac[1], Jupp = jac[2];
                for (int i = 0; i < k; i++) {
                    int globalRow = i * size + p;
                    for (int j = 0; j < k; j++) {
                        double fu   = Ju  [i][j];
                        double fup  = Jup [i][j];
                        double fupp = Jupp[i][j];
                        int colBlockOff = j * size;
                        for (int q = 0; q < size; q++) {
                            double d1 = scaledD [p * size + q];
                            double d2 = scaledD2[p * size + q];
                            double delta = (p == q) ? 1.0 : 0.0;
                            J[globalRow * total + (colBlockOff + q)] +=
                                fu * delta + fup * d1 + fupp * d2;
                        }
                    }
                }
            }
            // BC rows are linear — just the row coefficients.
            for (BcRow row : bcRows) {
                int rowOff = row.globalRow * total;
                for (int j = 0; j < total; j++) J[rowOff + j] = 0.0;
                for (int j = 0; j < total; j++) J[rowOff + j] = row.coeff[j];
            }

            // Solve J Δu = -r.
            double[] negR = new double[total];
            for (int i = 0; i < total; i++) negR[i] = -r[i];
            NDArray Jarr = new ConcreteNDArray(J, new Shape(total, total));
            NDArray rArr = new ConcreteNDArray(negR, new Shape(total));
            double[] du;
            try {
                du = Linalg.solve(Jarr, rArr).toDoubleArray();
            } catch (RuntimeException e) {
                return extractComponents(uFlat, k, size);
            }

            // Damped step with Armijo backtracking.
            double stepSize = damping;
            double bestNorm = Double.POSITIVE_INFINITY;
            double[] bestU = null;
            for (int back = 0; back <= MAX_BACKTRACKS; back++) {
                double[] uTry = uFlat.clone();
                for (int i = 0; i < total; i++) uTry[i] += stepSize * du[i];
                double normTry = evaluateResNorm(uTry, size, k, xGrid, scaledD, scaledD2, bcRows);
                if (normTry < bestNorm) { bestNorm = normTry; bestU = uTry; }
                if (normTry < resNorm) break;
                stepSize *= 0.5;
            }
            if (bestU == null) return extractComponents(uFlat, k, size);
            uFlat = bestU;

            double duNorm = ArrayMath.maxAbs(du) * stepSize;
            double uScale = Math.max(1.0, ArrayMath.maxAbs(uFlat));
            if (duNorm <= opts.tol() * uScale) return extractComponents(uFlat, k, size);
        }
        return extractComponents(uFlat, k, size);
    }

    private double evaluateResNorm(double[] uFlat, int size, int k, double[] xGrid,
                                   double[] scaledD, double[] scaledD2, BcRow[] bcRows) {
        double[][] u   = componentGrids(uFlat, k, size);
        double[][] up  = componentDerivatives(scaledD,  u, k, size);
        double[][] upp = componentDerivatives(scaledD2, u, k, size);
        double norm = 0.0;
        for (int p = 0; p < size; p++) {
            double[] uP   = column(u,   p, k);
            double[] upP  = column(up,  p, k);
            double[] uppP = column(upp, p, k);
            double[] fp = residual.at(xGrid[p], uP, upP, uppP);
            for (double v : fp) norm = Math.max(norm, Math.abs(v));
        }
        for (BcRow row : bcRows) {
            double d = row.dot(uFlat) - row.value;
            if (Math.abs(d) > norm) norm = Math.abs(d);
        }
        return norm;
    }

    /**
     * A single BC's linearization: the row coefficients over the flat
     * {@code u} vector, plus the RHS value.
     */
    private record BcRow(int globalRow, double[] coeff, double value) {
        double dot(double[] u) {
            double s = 0.0;
            for (int i = 0; i < u.length; i++) s += coeff[i] * u[i];
            return s;
        }
    }

    private BcRow[] buildBcRows(SystemBC[] bcA, SystemBC[] bcB, double[] scaledD,
                                int size, int total, int n, int k) {
        BcRow[] out = new BcRow[bcA.length + bcB.length];
        int[] topSlot = new int[k], botSlot = new int[k];
        int idx = 0;
        for (SystemBC bc : bcB) {
            int c = bc.component();
            int rowInBlock = topSlot[c]++;
            int globalRow = c * size + rowInBlock;
            out[idx++] = bcCoeffs(bc, globalRow, /*evalCol=*/0, scaledD, size, total);
        }
        for (SystemBC bc : bcA) {
            int c = bc.component();
            int rowFromBottom = botSlot[c]++;
            int rowInBlock = n - rowFromBottom;
            int globalRow = c * size + rowInBlock;
            out[idx++] = bcCoeffs(bc, globalRow, /*evalCol=*/n, scaledD, size, total);
        }
        return out;
    }

    private static BcRow bcCoeffs(SystemBC bc, int globalRow, int evalCol,
                                  double[] scaledD, int size, int total) {
        double[] coeff = new double[total];
        int colBlockOff = bc.component() * size;
        int evalOff = evalCol * size;
        double value;
        switch (bc) {
            case SystemBC.Dirichlet d -> {
                coeff[colBlockOff + evalCol] = 1.0;
                value = d.value();
            }
            case SystemBC.Neumann nb -> {
                for (int j = 0; j < size; j++) coeff[colBlockOff + j] = scaledD[evalOff + j];
                value = nb.value();
            }
            case SystemBC.Robin r -> {
                coeff[colBlockOff + evalCol] += r.alpha();
                for (int j = 0; j < size; j++) coeff[colBlockOff + j] += r.beta() * scaledD[evalOff + j];
                value = r.value();
            }
            default -> throw new IllegalStateException();
        }
        return new BcRow(globalRow, coeff, value);
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    private double[] scaledDMatrix(int n) {
        int size = n + 1;
        double[] D = DifferentiationMatrix.chebD(n);
        double jac = 2.0 / domain.length();
        double[] out = new double[size * size];
        for (int i = 0; i < size * size; i++) out[i] = jac * D[i];
        return out;
    }

    private double[] physicalGrid(int n) {
        double[] out = new double[n + 1];
        for (int j = 0; j <= n; j++) out[j] = domain.fromRef(Math.cos(Math.PI * j / n));
        return out;
    }

    private double[] seedInitialGuess(int size, int k, double[] xGrid,
                                      NewtonOptions opts, Chebfun[] previousSolution) {
        double[] uFlat = new double[k * size];
        if (previousSolution != null) {
            for (int c = 0; c < k; c++) {
                for (int p = 0; p < size; p++) uFlat[c * size + p] = previousSolution[c].feval(xGrid[p]);
            }
            return uFlat;
        }
        // No shared multi-component initial guess in NewtonOptions — the
        // scalar-guess field is repeated across every component when
        // present. Users who need distinct per-component guesses can
        // subclass or run a coarse solve first.
        Chebfun guess = opts.initialGuess();
        if (guess != null) {
            for (int c = 0; c < k; c++) {
                for (int p = 0; p < size; p++) uFlat[c * size + p] = guess.feval(xGrid[p]);
            }
        }
        return uFlat;
    }

    private static double[][] componentGrids(double[] uFlat, int k, int size) {
        double[][] out = new double[k][size];
        for (int c = 0; c < k; c++) System.arraycopy(uFlat, c * size, out[c], 0, size);
        return out;
    }

    private static double[][] componentDerivatives(double[] mat, double[][] u, int k, int size) {
        double[][] out = new double[k][size];
        for (int c = 0; c < k; c++) {
            for (int p = 0; p < size; p++) {
                double s = 0.0;
                for (int q = 0; q < size; q++) s += mat[p * size + q] * u[c][q];
                out[c][p] = s;
            }
        }
        return out;
    }

    private static double[] column(double[][] u, int p, int k) {
        double[] out = new double[k];
        for (int c = 0; c < k; c++) out[c] = u[c][p];
        return out;
    }

    private static double[][] extractComponents(double[] uFlat, int k, int size) {
        double[][] out = new double[k][size];
        for (int c = 0; c < k; c++) System.arraycopy(uFlat, c * size, out[c], 0, size);
        return out;
    }

}
