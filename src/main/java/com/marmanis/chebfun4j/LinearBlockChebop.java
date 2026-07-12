package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ArrayMath;
import com.marmanis.chebfun4j.util.DifferentiationMatrix;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * A linear <em>block</em> differential operator on a vector of unknowns
 * {@code u = [u_0, u_1, ..., u_{k-1}]}, each a {@link Chebfun} on a shared
 * {@link Domain}. The operator is built up as a sum of terms, each of
 * the form {@code L[i][j] += c(x) * d^n / dx^n} — a contribution to the
 * {@code i}-th equation from the {@code j}-th component's {@code n}-th
 * derivative.
 *
 * <p>{@link #solve} discretizes the coupled system {@code L u = f} on a
 * Chebyshev-2nd-kind grid via spectral collocation, replaces
 * {@code 2k} rows with the boundary conditions (see {@link SystemBC}),
 * and hands the resulting {@code k(n+1) × k(n+1)} linear system to
 * {@link Linalg#solve} — the same solver path {@link Chebop} uses,
 * scaled up to a block matrix. Adaptive: tries
 * {@code n = 16, 32, 64, ..., 512}, converges when the max change of
 * every component between successive grids drops below
 * {@code tol × vscale}.
 *
 * <p>MVP scope: constant-coefficient block entries with Dirichlet /
 * Neumann / Robin BCs. Variable-coefficient blocks (i.e. entries whose
 * coefficient is itself a {@link Chebfun}) are a small extension; the
 * scaffolding is already in place — {@link Term} carries a
 * {@link Chebfun} coefficient — but the adaptive loop's convergence
 * criterion doesn't yet vary the coefficient's discretization.
 */
public final class LinearBlockChebop {
    public static final int MAX_N = 512;
    public static final double DEFAULT_TOL = 1e-10;

    private static final int[] GRID_SIZES = {16, 32, 64, 128, 256, MAX_N};

    private record Term(int row, int col, int order, Chebfun coeff) {}

    private final Domain domain;
    private final int numComponents;
    private final List<Term> terms;

    private LinearBlockChebop(Domain domain, int numComponents, List<Term> terms) {
        this.domain = domain;
        this.numComponents = numComponents;
        this.terms = terms;
    }

    /** Start with the zero {@code k×k} block operator on {@code domain}. */
    public static LinearBlockChebop zero(Domain domain, int numComponents) {
        if (numComponents < 1) {
            throw new IllegalArgumentException("numComponents must be >= 1, got " + numComponents);
        }
        return new LinearBlockChebop(domain, numComponents, new ArrayList<>());
    }

    public Domain domain() { return domain; }
    public int numComponents() { return numComponents; }

    /**
     * Add the term {@code coeff * d^order u_col / dx^order} to the
     * {@code row}-th equation. Multiple terms with the same
     * {@code (row, col, order)} accumulate.
     */
    public LinearBlockChebop term(int row, int col, int order, double coeff) {
        return term(row, col, order, Chebfun.constant(coeff, domain));
    }

    /** Variable-coefficient version: {@code coeff(x) * d^order u_col / dx^order}. */
    public LinearBlockChebop term(int row, int col, int order, Chebfun coeff) {
        requireIndex(row, "row"); requireIndex(col, "col");
        if (order < 0) throw new IllegalArgumentException("order must be >= 0");
        if (!coeff.domain().equalsDomain(domain)) {
            throw new IllegalArgumentException(
                "coefficient chebfun must share the operator's domain");
        }
        List<Term> ts = new ArrayList<>(terms);
        ts.add(new Term(row, col, order, coeff));
        return new LinearBlockChebop(domain, numComponents, ts);
    }

    private void requireIndex(int idx, String label) {
        if (idx < 0 || idx >= numComponents) {
            throw new IllegalArgumentException(
                label + " index " + idx + " out of range [0, " + numComponents + ")");
        }
    }

    /**
     * Solve {@code L u = rhs} on the operator's domain with boundary
     * conditions {@code bcA} at {@code a} and {@code bcB} at {@code b}.
     * {@code rhs} must have length {@code numComponents}; the two BC
     * arrays' combined length is unconstrained here but must equal the
     * total number of degrees of freedom the BCs need to pin down (for a
     * standard 2nd-order {@code k}-component system that's {@code 2k}
     * conditions total — usually distributed {@code k}-per-endpoint).
     * The number of BC rows we overwrite in the collocation matrix
     * equals {@code bcA.length + bcB.length}.
     */
    public Chebfun[] solve(Chebfun[] rhs, SystemBC[] bcA, SystemBC[] bcB) {
        return solve(rhs, bcA, bcB, DEFAULT_TOL);
    }

    public Chebfun[] solve(Chebfun[] rhs, SystemBC[] bcA, SystemBC[] bcB, double tol) {
        if (rhs.length != numComponents) {
            throw new IllegalArgumentException(
                "rhs must have " + numComponents + " components, got " + rhs.length);
        }
        for (Chebfun f : rhs) {
            if (!f.domain().equalsDomain(domain)) {
                throw new IllegalArgumentException("rhs component must share the operator's domain");
            }
        }
        Chebfun[] last = null;
        for (int n : GRID_SIZES) {
            Chebfun[] candidate = solveAt(rhs, bcA, bcB, n);
            if (last != null) {
                double err = 0.0;
                double scale = 0.0;
                for (int c = 0; c < numComponents; c++) {
                    err = Math.max(err, ArrayMath.maxAbsDiff(candidate[c], last[c]));
                    scale = Math.max(scale, candidate[c].normInf());
                }
                if (err <= tol * Math.max(1.0, scale)) return candidate;
            }
            last = candidate;
        }
        return last;
    }

    private Chebfun[] solveAt(Chebfun[] rhs, SystemBC[] bcA, SystemBC[] bcB, int n) {
        int size = n + 1;
        int total = numComponents * size;

        double[] D = DifferentiationMatrix.chebD(n);
        double jac = 2.0 / domain.length();
        double[] scaledD = new double[size * size];
        for (int i = 0; i < size * size; i++) scaledD[i] = jac * D[i];

        // Powers of the scaled differentiation matrix up to the highest
        // order that appears in any term.
        int maxOrder = 0;
        for (Term t : terms) if (t.order > maxOrder) maxOrder = t.order;
        double[][] Dpow = new double[maxOrder + 1][];
        Dpow[0] = identity(size);
        if (maxOrder >= 1) Dpow[1] = scaledD;
        for (int p = 2; p <= maxOrder; p++) {
            Dpow[p] = DifferentiationMatrix.matMul(Dpow[p - 1], scaledD, size);
        }

        // Physical grid.
        double[] xGrid = new double[size];
        for (int j = 0; j <= n; j++) xGrid[j] = domain.fromRef(Math.cos(Math.PI * j / n));

        // Build the block matrix. Row block i, column block j occupies
        // rows [i*size, (i+1)*size) and columns [j*size, (j+1)*size). At
        // grid row p in block (i, j), the entry is the sum over terms
        // (i, j, ord) of coeff(x_p) * D^ord[p, q].
        double[] Ablock = new double[total * total];
        for (Term t : terms) {
            double[] Dk = Dpow[t.order];
            int rowBlockOff = t.row * size;
            int colBlockOff = t.col * size;
            for (int p = 0; p < size; p++) {
                double c = t.coeff.feval(xGrid[p]);
                if (c == 0.0) continue;
                for (int q = 0; q < size; q++) {
                    Ablock[(rowBlockOff + p) * total + (colBlockOff + q)] += c * Dk[p * size + q];
                }
            }
        }

        // RHS block vector.
        double[] rhsVec = new double[total];
        for (int i = 0; i < numComponents; i++) {
            for (int p = 0; p < size; p++) rhsVec[i * size + p] = rhs[i].feval(xGrid[p]);
        }

        // BC row replacement. Row 0 in each block corresponds to x = b,
        // row n to x = a — same ordering Chebop uses. We overwrite the
        // top rows for endpoint b's BCs and the bottom rows for
        // endpoint a's, distributing evenly across components as we go:
        // BC index i at endpoint b writes into block-row (i mod k)'s
        // "next available" top slot, and analogously at endpoint a.
        //
        // For the standard k-per-endpoint case with one BC per component
        // this reduces to "write BC at (component c, top row)" and "at
        // (component c, bottom row)", which is the natural choice.
        int[] topSlotUsed = new int[numComponents];
        int[] bottomSlotUsed = new int[numComponents];
        for (SystemBC bc : bcB) {
            int c = bc.component();
            int rowInBlock = topSlotUsed[c]++;
            int globalRow = c * size + rowInBlock;
            applySystemBc(Ablock, rhsVec, bc, globalRow, /*evalCol=*/0, scaledD, size, total);
        }
        for (SystemBC bc : bcA) {
            int c = bc.component();
            int rowFromBottom = bottomSlotUsed[c]++;
            int rowInBlock = n - rowFromBottom;
            int globalRow = c * size + rowInBlock;
            applySystemBc(Ablock, rhsVec, bc, globalRow, /*evalCol=*/n, scaledD, size, total);
        }

        // Solve the big block system.
        NDArray Aarr = new ConcreteNDArray(Ablock, new Shape(total, total));
        NDArray rArr = new ConcreteNDArray(rhsVec, new Shape(total));
        double[] uFlat = Linalg.solve(Aarr, rArr).toDoubleArray();

        // Reconstruct k Chebfuns from grid values.
        Chebfun[] result = new Chebfun[numComponents];
        for (int i = 0; i < numComponents; i++) {
            double[] uComp = new double[size];
            System.arraycopy(uFlat, i * size, uComp, 0, size);
            result[i] = new Chebfun(Chebtech.fromValues(uComp), domain);
        }
        return result;
    }

    /**
     * Overwrite {@code A[globalRow, *]} to enforce a single system BC on
     * component {@code bc.component()} at the evaluation point of the
     * corresponding endpoint. The BC contributes coefficients only in
     * the {@code component} block-column — cross-component BCs (like
     * {@code u_0(a) + u_1(a) = 5}) aren't in this MVP.
     */
    private static void applySystemBc(double[] A, double[] rhs, SystemBC bc,
                                      int globalRow, int evalCol,
                                      double[] scaledD, int size, int total) {
        for (int j = 0; j < total; j++) A[globalRow * total + j] = 0.0;
        int colBlockOff = bc.component() * size;
        int evalOff = evalCol * size;
        switch (bc) {
            case SystemBC.Dirichlet d -> {
                A[globalRow * total + (colBlockOff + evalCol)] = 1.0;
                rhs[globalRow] = d.value();
            }
            case SystemBC.Neumann nb -> {
                for (int j = 0; j < size; j++) {
                    A[globalRow * total + (colBlockOff + j)] = scaledD[evalOff + j];
                }
                rhs[globalRow] = nb.value();
            }
            case SystemBC.Robin r -> {
                A[globalRow * total + (colBlockOff + evalCol)] += r.alpha();
                for (int j = 0; j < size; j++) {
                    A[globalRow * total + (colBlockOff + j)] += r.beta() * scaledD[evalOff + j];
                }
                rhs[globalRow] = r.value();
            }
        }
    }

    private static double[] identity(int size) {
        double[] I = new double[size * size];
        for (int i = 0; i < size; i++) I[i * size + i] = 1.0;
        return I;
    }

}
