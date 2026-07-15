package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ArrayMath;
import com.marmanis.chebfun4j.util.DifferentiationMatrix;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 * A linear differential operator on {@code Chebfun}s over a {@link Domain},
 * built as a sum of terms {@code c_k(x) * d^k/dx^k}. Solves the boundary-
 * value problem {@code L u = f} on {@code [a, b]} with Dirichlet
 * conditions {@code u(a) = alpha, u(b) = beta} via Chebyshev spectral
 * collocation (Trefethen, <em>Spectral Methods in MATLAB</em>): enforce
 * the ODE at the Chebyshev-2nd-kind grid points, replace the top and
 * bottom rows with the BCs, and solve the resulting
 * {@code (n+1) x (n+1)} linear system with
 * {@link com.marmanis.jax4j.api.Linalg#solve}.
 *
 * <p>Adaptive: try {@code n = 16, 32, ..., 512} until successive solutions
 * agree at their shared grid points to within {@code tol * ||u||_inf}, or
 * cap at {@code 512}. Each candidate {@code n} is picked so that
 * {@code n + 1} lands on a valid {@code 2^k + 1} Chebtech length; no
 * resampling is needed to hand the values off to
 * {@link Chebtech#fromValues}.
 *
 * <p>MVP scope: single scalar linear ODE, constant or Chebfun coefficients
 * on each derivative, Dirichlet BCs at both endpoints. Nonlinear ODEs,
 * eigenvalue problems, and Neumann/Robin BCs are follow-ups.
 */
public final class Chebop {
    /** Maximum grid size the adaptive loop tries before giving up. */
    public static final int MAX_N = 512;

    /** Convergence tolerance (max change at shared points between successive n). */
    public static final double DEFAULT_TOL = 1e-10;

    private static final int[] GRID_SIZES = {16, 32, 64, 128, 256, MAX_N};

    private record Term(int order, Chebfun coeff) {}

    private final Domain domain;
    private final List<Term> terms;

    private Chebop(Domain domain, List<Term> terms) {
        this.domain = domain;
        this.terms = terms;
    }

    /** Start with the zero operator on {@code domain}. Add terms with {@link #plus}. */
    public static Chebop zero(Domain domain) {
        return new Chebop(domain, new ArrayList<>());
    }

    public Domain domain() { return domain; }

    /**
     * Convenience factory for a linear ODE with constant coefficients:
     * {@code coeffs[0] u + coeffs[1] u' + coeffs[2] u'' + ...}.
     */
    public static Chebop constantCoefficients(Domain domain, double... coeffs) {
        Chebop op = zero(domain);
        for (int k = 0; k < coeffs.length; k++) {
            if (coeffs[k] != 0.0) op = op.plus(k, coeffs[k]);
        }
        return op;
    }

    /** Add a term {@code coeff * d^order u / dx^order} (constant coefficient). */
    public Chebop plus(int order, double coeff) {
        return plus(order, Chebfun.constant(coeff, domain));
    }

    /** Add a variable-coefficient term. */
    public Chebop plus(int order, Chebfun coeff) {
        if (order < 0) throw new IllegalArgumentException("derivative order must be >= 0");
        if (!coeff.domain().equalsDomain(domain)) {
            throw new IllegalArgumentException(
                "coefficient chebfun must share the operator's domain");
        }
        List<Term> ts = new ArrayList<>(terms);
        ts.add(new Term(order, coeff));
        return new Chebop(domain, ts);
    }

    /**
     * Solve {@code L u = rhs(x)} with Dirichlet BCs {@code u(a) = alpha},
     * {@code u(b) = beta} — a convenience wrapper around
     * {@link #solve(Chebfun, BoundaryCondition, BoundaryCondition)}.
     */
    public Chebfun solve(Chebfun rhs, double alpha, double beta) {
        return solve(rhs, new BoundaryCondition.Dirichlet(alpha),
                          new BoundaryCondition.Dirichlet(beta));
    }

    /** Same as {@link #solve(Chebfun, double, double)} with a raw {@code f(x)} RHS. */
    public Chebfun solve(DoubleUnaryOperator rhs, double alpha, double beta) {
        return solve(rhs, new BoundaryCondition.Dirichlet(alpha),
                          new BoundaryCondition.Dirichlet(beta), DEFAULT_TOL);
    }

    /**
     * Solve {@code L u = rhs} with the given boundary conditions at {@code a}
     * and {@code b}. Each BC may be {@code Dirichlet}, {@code Neumann}, or
     * {@code Robin} (see {@link BoundaryCondition}).
     */
    public Chebfun solve(Chebfun rhs, BoundaryCondition bcA, BoundaryCondition bcB) {
        return solve(rhs::feval, bcA, bcB, DEFAULT_TOL);
    }

    public Chebfun solve(DoubleUnaryOperator rhs, BoundaryCondition bcA, BoundaryCondition bcB) {
        return solve(rhs, bcA, bcB, DEFAULT_TOL);
    }

    public Chebfun solve(DoubleUnaryOperator rhs,
                         BoundaryCondition bcA, BoundaryCondition bcB,
                         double tol) {
        Chebfun last = null;
        for (int n : GRID_SIZES) {
            Chebfun candidate = discretize(bcA, bcB, n).solve(rhs);
            if (last != null) {
                double vscale = Math.max(1.0, candidate.normInf());
                // Exact residual via difference-normInf, not the sampled probe
                // used before. The extra Chebfun arithmetic is dwarfed by the
                // O(n³) solves themselves.
                double err = candidate.minus(last).normInf();
                if (err <= tol * vscale) return candidate;
            }
            last = candidate;
        }
        return last;
    }

    /**
     * A cached, LU-factored discretisation of this operator at grid size
     * {@code n} with the given boundary conditions. Once obtained, a
     * {@link #solve} call is O(n²) — the O(n³) matrix build + factorisation
     * is paid only once. Use for parametric sweeps in the right-hand side,
     * batch solves, or when you want to interrogate the discretised system
     * (e.g. via {@link #conditionNumber}).
     */
    public static final class Discretization {
        private final int n;
        private final double[] xGrid;
        private final Linalg.LU lu;
        private final double bcAValue;
        private final double bcBValue;
        private final Domain domain;

        private Discretization(int n, double[] xGrid, Linalg.LU lu,
                               double bcAValue, double bcBValue, Domain domain) {
            this.n = n;
            this.xGrid = xGrid;
            this.lu = lu;
            this.bcAValue = bcAValue;
            this.bcBValue = bcBValue;
            this.domain = domain;
        }

        /** Grid size the operator was discretised at. */
        public int n() { return n; }

        /**
         * Solve {@code L u = rhs} at this fixed grid, returning the smooth
         * Chebfun that interpolates the collocation-grid values.
         */
        public Chebfun solve(DoubleUnaryOperator rhs) {
            return new Chebfun(Chebtech.fromValues(solveRawValues(rhs)), domain);
        }

        /** Same as {@link #solve(DoubleUnaryOperator)} for a Chebfun RHS. */
        public Chebfun solve(Chebfun rhs) {
            return solve(rhs::feval);
        }

        /**
         * ∞-norm condition-number estimate of the discretised collocation
         * matrix (with the BC rows already substituted in). A large value
         * — say, {@code > 10¹⁰} — means the linear solve at this grid is
         * losing significant digits and the reported solution deserves
         * scepticism. Uses {@link Linalg#cond(Linalg.LU)}: Hager's iteration
         * against the already-cached factorisation, so no re-factorisation.
         */
        public double conditionNumber() {
            return Linalg.cond(lu);
        }

        /**
         * Raw collocation-grid values solving {@code L u = rhs}. Used by
         * {@link Chebop#solveAt} and by {@link Chebop#solve}'s adaptive
         * loop; end users typically want the {@link Chebfun} form.
         */
        double[] solveRawValues(DoubleUnaryOperator rhs) {
            int size = n + 1;
            double[] b = new double[size];
            for (int i = 0; i < size; i++) b[i] = rhs.applyAsDouble(xGrid[i]);
            // BC rows override RHS at row 0 (x = b endpoint) and row n (x = a).
            b[0] = bcBValue;
            b[n] = bcAValue;
            return lu.solve(b);
        }
    }

    /**
     * Discretise this operator to an {@code (n+1) × (n+1)} spectral
     * collocation system with the given boundary conditions, factor it,
     * and return the reusable {@link Discretization}. Every subsequent
     * {@link Discretization#solve} is O(n²).
     */
    public Discretization discretize(BoundaryCondition bcA, BoundaryCondition bcB, int n) {
        int size = n + 1;
        double[] D = DifferentiationMatrix.chebD(n);
        // Scale from reference [-1, 1] to physical [a, b]: D_physical = (2/L) D_ref.
        double jac = 2.0 / domain.length();
        double[] scaledD = new double[size * size];
        for (int i = 0; i < size * size; i++) scaledD[i] = jac * D[i];

        int maxOrder = 0;
        for (Term t : terms) if (t.order() > maxOrder) maxOrder = t.order();
        double[][] Dpow = new double[maxOrder + 1][];
        Dpow[0] = identity(size);
        if (maxOrder >= 1) Dpow[1] = scaledD;
        for (int k = 2; k <= maxOrder; k++) {
            Dpow[k] = DifferentiationMatrix.matMul(Dpow[k - 1], scaledD, size);
        }

        // Chebyshev-2nd-kind grid on the physical domain, ordered so index 0
        // corresponds to y = +1 (x = b) and index n to y = -1 (x = a) — the
        // ordering Chebtech.fromValues expects.
        double[] xGrid = new double[size];
        for (int j = 0; j <= n; j++) xGrid[j] = domain.fromRef(Math.cos(Math.PI * j / n));

        double[] L = new double[size * size];
        for (Term t : terms) {
            double[] Dk = Dpow[t.order()];
            for (int i = 0; i < size; i++) {
                double c = t.coeff().feval(xGrid[i]);
                if (c == 0.0) continue;
                for (int j = 0; j < size; j++) L[i * size + j] += c * Dk[i * size + j];
            }
        }

        // Overwrite the two BC rows with their linear-form coefficients.
        applyBcToMatrix(L, bcB, /*row=*/0, /*evalCol=*/0, scaledD, size);
        applyBcToMatrix(L, bcA, /*row=*/n, /*evalCol=*/n, scaledD, size);

        Linalg.LU lu = Linalg.lu(new ConcreteNDArray(L, new Shape(size, size)));
        return new Discretization(n, xGrid, lu, bcA.value(), bcB.value(), domain);
    }

    /**
     * @deprecated Prefer the {@link BoundaryCondition}-based overload.
     */
    @Deprecated
    public double[] solveAt(DoubleUnaryOperator rhs, double alpha, double beta, int n) {
        return solveAt(rhs, new BoundaryCondition.Dirichlet(alpha),
                            new BoundaryCondition.Dirichlet(beta), n);
    }

    /** Solve at fixed grid size {@code n} — advanced entry point. */
    public double[] solveAt(DoubleUnaryOperator rhs,
                            BoundaryCondition bcA, BoundaryCondition bcB, int n) {
        return discretize(bcA, bcB, n).solveRawValues(rhs);
    }

    /**
     * Overwrite {@code L[row, *]} to enforce the given boundary condition's
     * linear form at the collocation point of grid index {@code evalCol} —
     * i.e. the endpoint where the BC is being imposed. Dirichlet
     * contributes an identity row selecting that grid value; Neumann
     * contributes the {@code evalCol}-th row of the scaled differentiation
     * matrix (a discrete derivative at that point); Robin is the
     * {@code alpha}/{@code beta} linear combination of the two. The BC's
     * scalar value (right-hand side) is handled separately by the caller.
     */
    private static void applyBcToMatrix(double[] L, BoundaryCondition bc,
                                        int row, int evalCol, double[] scaledD, int size) {
        int rowOff = row * size;
        for (int j = 0; j < size; j++) L[rowOff + j] = 0.0;
        int evalOff = evalCol * size;
        switch (bc) {
            case BoundaryCondition.Dirichlet d -> L[rowOff + evalCol] = 1.0;
            case BoundaryCondition.Neumann nb -> {
                for (int j = 0; j < size; j++) L[rowOff + j] = scaledD[evalOff + j];
            }
            case BoundaryCondition.Robin r -> {
                L[rowOff + evalCol] += r.alpha();
                for (int j = 0; j < size; j++) L[rowOff + j] += r.beta() * scaledD[evalOff + j];
            }
        }
    }

    /**
     * Overwrite {@code L[row, *]} and {@code b[row]} for the given BC.
     * Used by {@link #eigsAt} where the "b" is throwaway scratch anyway;
     * the {@link #discretize} path uses {@link #applyBcToMatrix} plus
     * separately stashing {@code bc.value()} on the {@link Discretization}.
     */
    private static void applyBc(double[] L, double[] b, BoundaryCondition bc,
                                int row, int evalCol, double[] scaledD, int size) {
        applyBcToMatrix(L, bc, row, evalCol, scaledD, size);
        b[row] = bc.value();
    }

    /**
     * A generalized eigenvalue and its corresponding eigenfunction on the
     * operator's domain. Returned in bulk by {@link #eigs}.
     */
    public record Eigs(double[] eigenvalues, Quasimatrix eigenfunctions) {}

    /**
     * Compute the {@code k} smallest-real-part generalized eigenvalues and
     * corresponding eigenfunctions of the operator {@code L} subject to
     * boundary conditions {@code bcA} at {@code a} and {@code bcB} at
     * {@code b}. Discretizes to a spectral-collocation matrix
     * {@code L u = lambda B u} where {@code B} is the identity mass matrix
     * with the BC rows zeroed (so the BCs don't add spurious eigenvalues),
     * calls {@link com.marmanis.jax4j.api.Linalg#eig(NDArray, NDArray)},
     * filters real eigenvalues, and takes the {@code k} smallest by real
     * part.
     *
     * <p>Eigenfunctions are returned {@code L^2}-normalized in a
     * {@link Quasimatrix} whose columns are aligned with the
     * eigenvalues array.
     *
     * <p>Adaptive: tries {@code n = 16, 32, 64, ...} until the first
     * {@code k} eigenvalues stabilize between successive discretizations.
     */
    public Eigs eigs(BoundaryCondition bcA, BoundaryCondition bcB, int k) {
        return eigs(bcA, bcB, k, DEFAULT_TOL);
    }

    public Eigs eigs(BoundaryCondition bcA, BoundaryCondition bcB, int k, double tol) {
        double[] prevValues = null;
        Eigs last = null;
        for (int n : GRID_SIZES) {
            Eigs candidate = eigsAt(bcA, bcB, k, n);
            if (prevValues != null) {
                double err = ArrayMath.maxAbsDiff(candidate.eigenvalues(), prevValues, k);
                double scale = Math.max(1.0, ArrayMath.maxAbs(candidate.eigenvalues()));
                if (err <= tol * scale) return candidate;
            }
            prevValues = candidate.eigenvalues();
            last = candidate;
        }
        return last;
    }

    private Eigs eigsAt(BoundaryCondition bcA, BoundaryCondition bcB, int k, int n) {
        int size = n + 1;
        double[] D = DifferentiationMatrix.chebD(n);
        double jac = 2.0 / domain.length();
        double[] scaledD = new double[size * size];
        for (int i = 0; i < size * size; i++) scaledD[i] = jac * D[i];

        int maxOrder = 0;
        for (Term t : terms) if (t.order() > maxOrder) maxOrder = t.order();
        double[][] Dpow = new double[maxOrder + 1][];
        Dpow[0] = identity(size);
        if (maxOrder >= 1) Dpow[1] = scaledD;
        for (int p = 2; p <= maxOrder; p++) {
            Dpow[p] = DifferentiationMatrix.matMul(Dpow[p - 1], scaledD, size);
        }

        double[] xGrid = new double[size];
        for (int j = 0; j <= n; j++) xGrid[j] = domain.fromRef(Math.cos(Math.PI * j / n));

        double[] Lmat = new double[size * size];
        for (Term t : terms) {
            double[] Dp = Dpow[t.order()];
            for (int i = 0; i < size; i++) {
                double c = t.coeff().feval(xGrid[i]);
                if (c == 0.0) continue;
                for (int j = 0; j < size; j++) Lmat[i * size + j] += c * Dp[i * size + j];
            }
        }

        // Homogeneous BCs (non-zero BC values are ignored — the eigenvalue
        // problem is inherently homogeneous; we keep the same BC type as
        // solve() for API symmetry). Build BC rows into a scratch matrix
        // so we can extract their coefficients on u_0, u_n, and interior
        // u_1..u_{n-1}, then solve a 2x2 system to express (u_0, u_n) as
        // linear combinations of the interior — which reduces the
        // eigenvalue problem to a standard one on n-1 unknowns without
        // needing a mass matrix.
        //
        // For Dirichlet-Dirichlet this collapses to u_0 = u_n = 0 and the
        // reduction just drops rows/columns 0 and n. For Neumann/Robin the
        // BC row contributes real coefficients that get absorbed into the
        // interior rows.
        int m = n - 1;
        if (m < 1) throw new IllegalArgumentException("eigs requires grid size n > 1");
        double[] LbcRow0 = new double[size];
        double[] LbcRowN = new double[size];
        double[] zeroRhs = new double[size];
        // Reuse applyBc to fill single rows.
        double[] scratch = new double[size * size];
        applyBc(scratch, zeroRhs, bcB, /*row=*/0, /*evalCol=*/0, scaledD, size);
        applyBc(scratch, zeroRhs, bcA, /*row=*/n, /*evalCol=*/n, scaledD, size);
        for (int j = 0; j < size; j++) {
            LbcRow0[j] = scratch[0 * size + j];
            LbcRowN[j] = scratch[n * size + j];
        }
        double Aco = LbcRow0[0], Bco = LbcRow0[n];
        double Cco = LbcRowN[0], Dco = LbcRowN[n];
        double det = Aco * Dco - Bco * Cco;
        if (Math.abs(det) < 1e-14 * Math.max(1.0, Math.max(Math.abs(Aco) + Math.abs(Bco),
                                                            Math.abs(Cco) + Math.abs(Dco)))) {
            throw new IllegalStateException(
                "eigs: BC rows do not uniquely determine boundary DOFs (2x2 BC det = " + det + ")");
        }
        double[] alpha = new double[m]; // u_0 = sum alpha_j u_j (interior)
        double[] gamma = new double[m]; // u_n = sum gamma_j u_j (interior)
        for (int j = 1; j <= n - 1; j++) {
            double pj = LbcRow0[j];
            double qj = LbcRowN[j];
            alpha[j - 1] = (-Dco * pj + Bco * qj) / det;
            gamma[j - 1] = (Cco * pj - Aco * qj) / det;
        }
        // Reduced interior matrix: L_red[i][j] = L[i+1][j+1]
        //   + L[i+1][0] * alpha[j] + L[i+1][n] * gamma[j].
        double[] Linterior = new double[m * m];
        for (int i = 0; i < m; i++) {
            int ir = (i + 1) * size;
            double li0 = Lmat[ir + 0];
            double liN = Lmat[ir + n];
            for (int j = 0; j < m; j++) {
                Linterior[i * m + j] =
                    Lmat[ir + (j + 1)] + li0 * alpha[j] + liN * gamma[j];
            }
        }

        NDArray A = new ConcreteNDArray(Linterior, new Shape(m, m));
        NDArray[] w = Linalg.eig(A);
        double[] wr = w[0].toDoubleArray();
        double[] wi = w[1].toDoubleArray();

        // Filter for real, finite eigenvalues; drop tiny imaginary residues.
        java.util.List<Integer> keep = new java.util.ArrayList<>();
        double scaleGuess = 0.0;
        for (double v : wr) if (Double.isFinite(v)) scaleGuess = Math.max(scaleGuess, Math.abs(v));
        double imTol = 1e-6 * Math.max(1.0, scaleGuess);
        for (int i = 0; i < wr.length; i++) {
            if (!Double.isFinite(wr[i])) continue;
            if (Math.abs(wi[i]) > imTol) continue;
            keep.add(i);
        }
        keep.sort(java.util.Comparator.comparingDouble(i -> wr[i]));
        int take = Math.min(k, keep.size());
        double[] vals = new double[take];
        for (int i = 0; i < take; i++) vals[i] = wr[keep.get(i)];

        // Extract eigenvectors by inverse iteration on the interior system,
        // then reconstruct u_0, u_n from the (alpha, gamma) coefficients
        // that expressed the boundary DOFs in terms of the interior. For
        // Dirichlet BCs, alpha and gamma are identically zero and this
        // reduces to just zero-padding the boundary positions.
        Chebfun[] eigfuns = new Chebfun[take];
        double[] Bidentity = identity(m);
        for (int i = 0; i < take; i++) {
            double lambda = vals[i];
            double[] interiorEigVec = inverseIteration(Linterior, Bidentity, m, lambda, /*seed=*/i + 7);
            double[] fullEigVec = new double[size];
            for (int j = 0; j < m; j++) fullEigVec[j + 1] = interiorEigVec[j];
            double u0 = 0.0, un = 0.0;
            for (int j = 0; j < m; j++) {
                u0 += alpha[j] * interiorEigVec[j];
                un += gamma[j] * interiorEigVec[j];
            }
            fullEigVec[0] = u0;
            fullEigVec[n] = un;
            eigfuns[i] = new Chebfun(Chebtech.fromValues(fullEigVec), domain);
        }
        Quasimatrix eigQmat = new Quasimatrix(eigfuns).normalizeColumns();
        return new Eigs(vals, eigQmat);
    }

    /**
     * Extract an eigenvector of the pencil {@code (L, B)} at the known
     * eigenvalue {@code lambda} by inverse iteration: solve
     * {@code (L - lambda B) v = w} for {@code w} a random seed vector,
     * normalize, repeat. A shift-and-invert step converges rapidly once
     * {@code lambda} is close to a true eigenvalue.
     *
     * <p>The shifted matrix is constant across the 20 iterations, so we
     * factor it once via {@link Linalg#lu} and back-substitute each iter
     * — O(n³) once + O(n²) × 20, versus the old O(n³) × 20.
     */
    private static double[] inverseIteration(double[] L, double[] B, int size,
                                             double lambda, long seed) {
        double[] v = new double[size];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < size; i++) v[i] = rng.nextGaussian();
        // Nudge lambda slightly to avoid an exactly-singular shift-and-invert
        // step; the true eigenvalue was returned by a fallible float
        // computation so we can trust it only to a few ulps of scaled ε.
        double nudge = 1e-10 * Math.max(1.0, Math.abs(lambda));
        double[] Mn = new double[size * size];
        for (int i = 0; i < size * size; i++) Mn[i] = L[i] - lambda * B[i];
        for (int i = 0; i < size; i++) Mn[i * size + i] -= nudge;

        Linalg.LU lu;
        try {
            lu = Linalg.lu(new ConcreteNDArray(Mn, new Shape(size, size)));
        } catch (RuntimeException e) {
            // Very close to an exact eigenvalue: factorisation fails. The
            // random seed vector is our best guess.
            return v;
        }
        for (int iter = 0; iter < 20; iter++) {
            double[] next = lu.solve(v);
            double norm = 0.0;
            for (double x : next) norm += x * x;
            norm = Math.sqrt(norm);
            if (norm == 0.0) break;
            for (int i = 0; i < size; i++) next[i] /= norm;
            v = next;
        }
        return v;
    }

    private static double[] identity(int size) {
        double[] I = new double[size * size];
        for (int i = 0; i < size; i++) I[i * size + i] = 1.0;
        return I;
    }
}
