package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ChebyshevPoints;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleBinaryOperator;

/**
 * A smooth real bivariate function on a rectangle {@code [xa, xb] × [ya, yb]},
 * stored as a <em>low-rank</em> Chebyshev approximation
 * <pre>
 *   f(x, y) = sum_{i=0}^{r-1}  cols_i(x) * rows_i(y) / pivotValues_i
 * </pre>
 * where each {@code cols_i} is a 1-D {@link Chebfun} in {@code x} on
 * {@code [xa, xb]} and each {@code rows_i} a 1-D {@link Chebfun} in
 * {@code y} on {@code [ya, yb]}. This is the same layout MATLAB
 * chebfun2 uses; a smooth 2-D function typically has a very small
 * separation rank (e.g. {@code sin(x) cos(y)} is rank 1), which is why
 * this representation is so much more compact than the full 2-D
 * Chebyshev coefficient tensor.
 *
 * <p>The adaptive constructor uses <b>Adaptive Cross Approximation
 * (ACA)</b> with complete pivoting on a sample grid: find the sample of
 * the current residual with maximum absolute value, extract that
 * column and row as 1-D chebfuns of the full function values there, add
 * the resulting rank-1 term to the approximation, subtract from the
 * residual, and repeat until the residual peak falls below
 * {@code tol * vscale}. Grid resolution doubles when the peak stops
 * decreasing, so functions of high separation rank still resolve.
 *
 * <p>MVP scope: single smooth piece over one rectangle. Piecewise
 * chebfun2 (splitting-on) and singular-edge chebfun2 are follow-ups.
 */
public final class Chebfun2 {
    /** Relative tolerance for the ACA convergence check. */
    public static final double DEFAULT_TOL = 1e-13;

    /** Grid sizes to try during construction. Each is a valid Chebtech length. */
    private static final int[] GRID_SIZES = {17, 33, 65, 129, 257};

    /** Maximum ACA rank per grid attempt before we escalate the grid. */
    private static final int MAX_RANK_PER_GRID = 100;

    private final Chebfun[] cols;
    private final Chebfun[] rows;
    private final double[] pivotValues;
    private final Rectangle rect;

    Chebfun2(Chebfun[] cols, Chebfun[] rows, double[] pivotValues, Rectangle rect) {
        if (cols.length != rows.length || cols.length != pivotValues.length) {
            throw new IllegalArgumentException(
                "cols, rows, and pivotValues must have the same length");
        }
        this.cols = cols;
        this.rows = rows;
        this.pivotValues = pivotValues;
        this.rect = rect;
    }

    /**
     * Adaptively construct a chebfun2 of {@code f(x, y)} on {@code rect} to
     * approximately {@link #DEFAULT_TOL} relative precision.
     */
    public Chebfun2(DoubleBinaryOperator f, Rectangle rect) {
        this(f, rect, DEFAULT_TOL);
    }

    public Chebfun2(DoubleBinaryOperator f, Rectangle rect, double tol) {
        this.rect = rect;
        // Try each grid size in turn. On each grid, build up rank-1 terms
        // by ACA until either the residual peak drops below tol*vscale
        // (success), or we hit the per-grid rank cap without converging
        // (escalate). At the largest grid we accept the best we have.
        Chebfun[] finalCols = null;
        Chebfun[] finalRows = null;
        double[] finalPivots = null;
        for (int gi = 0; gi < GRID_SIZES.length; gi++) {
            int n = GRID_SIZES[gi];
            Result r = acaOnGrid(f, rect, n, tol);
            finalCols = r.cols;
            finalRows = r.rows;
            finalPivots = r.pivots;
            if (r.converged) break;
        }
        this.cols = finalCols;
        this.rows = finalRows;
        this.pivotValues = finalPivots;
    }

    private record Result(Chebfun[] cols, Chebfun[] rows, double[] pivots, boolean converged) {}

    private static Result acaOnGrid(DoubleBinaryOperator f, Rectangle rect, int n, double tol) {
        Domain xDom = rect.xDomain();
        Domain yDom = rect.yDomain();
        double[] xRef = ChebyshevPoints.secondKind(n);
        double[] yRef = ChebyshevPoints.secondKind(n);
        double[] xPhys = new double[n];
        double[] yPhys = new double[n];
        for (int i = 0; i < n; i++) xPhys[i] = xDom.fromRef(xRef[i]);
        for (int i = 0; i < n; i++) yPhys[i] = yDom.fromRef(yRef[i]);
        // Residual matrix R[i][j] = f(xPhys[i], yPhys[j]) initially.
        double[][] R = new double[n][n];
        double vscale = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double v = f.applyAsDouble(xPhys[i], yPhys[j]);
                R[i][j] = v;
                double a = Math.abs(v);
                if (a > vscale) vscale = a;
            }
        }
        if (vscale == 0.0) vscale = 1.0;
        double cutoff = tol * vscale;

        List<Chebfun> colList = new ArrayList<>();
        List<Chebfun> rowList = new ArrayList<>();
        List<Double> pivotList = new ArrayList<>();

        boolean converged = false;
        for (int step = 0; step < MAX_RANK_PER_GRID; step++) {
            // Find max-abs entry of R.
            int pi = 0, pj = 0;
            double peak = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double a = Math.abs(R[i][j]);
                    if (a > peak) { peak = a; pi = i; pj = j; }
                }
            }
            if (peak <= cutoff) { converged = true; break; }
            double pivot = R[pi][pj];
            // Extract the column R[:, pj] and row R[pi, :] as 1-D chebtechs.
            // The values sit on the Chebyshev-2nd-kind grid in the physical
            // domain, and Chebtech.fromValues just needs them in reference-
            // grid order, which is already the case.
            // Snapshot the pivot column and row BEFORE the rank-1 update —
            // reading them straight out of R inside the update loop is a bug
            // because the very first inner iteration zeroes R[pi][:] and
            // R[:][pj], so subsequent iterations subtract garbage.
            double[] colVals = new double[n];
            double[] rowVals = new double[n];
            for (int i = 0; i < n; i++) colVals[i] = R[i][pj];
            for (int j = 0; j < n; j++) rowVals[j] = R[pi][j];
            Chebfun colFn = new Chebfun(Chebtech.fromValues(colVals).simplify(tol), xDom);
            Chebfun rowFn = new Chebfun(Chebtech.fromValues(rowVals).simplify(tol), yDom);
            colList.add(colFn);
            rowList.add(rowFn);
            pivotList.add(pivot);
            // Subtract R -= (colVals outer rowVals) / pivot.
            double invPivot = 1.0 / pivot;
            for (int i = 0; i < n; i++) {
                double ci = colVals[i] * invPivot;
                if (ci == 0.0) continue;
                for (int j = 0; j < n; j++) {
                    R[i][j] -= ci * rowVals[j];
                }
            }
        }
        Chebfun[] cA = colList.toArray(new Chebfun[0]);
        Chebfun[] rA = rowList.toArray(new Chebfun[0]);
        double[] pA = new double[pivotList.size()];
        for (int i = 0; i < pA.length; i++) pA[i] = pivotList.get(i);
        return new Result(cA, rA, pA, converged);
    }

    /** The bounding rectangle. */
    public Rectangle rectangle() { return rect; }

    /** Separation rank — number of rank-1 outer products. */
    public int rank() { return cols.length; }

    /**
     * The {@code i}-th column chebfun in {@code x} (a slice of {@code f}
     * along {@code y}, up to the pivot scaling).
     */
    public Chebfun col(int i) { return cols[i]; }

    /**
     * The {@code i}-th row chebfun in {@code y} (a slice of {@code f}
     * along {@code x}).
     */
    public Chebfun row(int i) { return rows[i]; }

    /** The pivot value {@code f(x_i*, y_i*)} at the {@code i}-th ACA step. */
    public double pivotValue(int i) { return pivotValues[i]; }

    /**
     * Evaluate {@code f(x, y)} for {@code (x, y)} in the rectangle. Values
     * for points outside the rectangle are extrapolated by the underlying
     * Chebyshev polynomials — usable near an edge but degrades quickly.
     */
    public double feval(double x, double y) {
        double sum = 0.0;
        for (int i = 0; i < cols.length; i++) {
            sum += cols[i].feval(x) * rows[i].feval(y) / pivotValues[i];
        }
        return sum;
    }

    /**
     * Double integral {@code integral_ya^yb integral_xa^xb f(x, y) dx dy}.
     * Uses the separable structure: each rank-1 term contributes
     * {@code cols[i].sum() * rows[i].sum() / pivotValues[i]}.
     */
    public double sum2() {
        double total = 0.0;
        for (int i = 0; i < cols.length; i++) {
            total += cols[i].sum() * rows[i].sum() / pivotValues[i];
        }
        return total;
    }

    /**
     * Marginal integral along one axis, returning a 1-D chebfun in the
     * other. {@code axis == 0} integrates over {@code x}, leaving a
     * function of {@code y}; {@code axis == 1} integrates over {@code y}
     * leaving a function of {@code x}.
     */
    public Chebfun sum(int axis) {
        if (axis == 0) {
            // integrate over x: sum_i cols[i].sum() * rows[i] / pivotValues[i].
            Chebfun result = null;
            for (int i = 0; i < cols.length; i++) {
                double scale = cols[i].sum() / pivotValues[i];
                Chebfun term = rows[i].times(scale);
                result = (result == null) ? term : result.plus(term);
            }
            return result;
        } else if (axis == 1) {
            Chebfun result = null;
            for (int i = 0; i < cols.length; i++) {
                double scale = rows[i].sum() / pivotValues[i];
                Chebfun term = cols[i].times(scale);
                result = (result == null) ? term : result.plus(term);
            }
            return result;
        } else {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
    }

    /**
     * Partial derivative {@code df/dx}: differentiate the {@code x}-factor
     * of every rank-1 term, keep the rest.
     */
    public Chebfun2 partialX() {
        Chebfun[] newCols = new Chebfun[cols.length];
        for (int i = 0; i < cols.length; i++) newCols[i] = cols[i].diff();
        return new Chebfun2(newCols, rows.clone(), pivotValues.clone(), rect);
    }

    /** Partial derivative {@code df/dy}. */
    public Chebfun2 partialY() {
        Chebfun[] newRows = new Chebfun[rows.length];
        for (int i = 0; i < rows.length; i++) newRows[i] = rows[i].diff();
        return new Chebfun2(cols.clone(), newRows, pivotValues.clone(), rect);
    }

    /**
     * Scalar multiply — absorbs into the pivot values (which are in the
     * denominator, so we divide) or equivalently could scale the cols /
     * rows; we choose the pivot form so the rank-1 factors are unchanged.
     */
    public Chebfun2 times(double s) {
        double[] newPiv = new double[pivotValues.length];
        for (int i = 0; i < pivotValues.length; i++) newPiv[i] = pivotValues[i] / s;
        return new Chebfun2(cols.clone(), rows.clone(), newPiv, rect);
    }

    /**
     * Add two chebfun2's on the same rectangle. The result has rank at
     * most the sum of the ranks — the two lists of rank-1 terms are
     * simply concatenated. Compression (dropping negligible terms) is a
     * follow-up; for MVP the rank grows monotonically.
     */
    public Chebfun2 plus(Chebfun2 other) {
        requireSameRect(other, "plus");
        return concatenate(other, +1.0);
    }

    /** Subtract two chebfun2's on the same rectangle. */
    public Chebfun2 minus(Chebfun2 other) {
        requireSameRect(other, "minus");
        return concatenate(other, -1.0);
    }

    /**
     * Pointwise product {@code (f * g)(x, y) = f(x, y) * g(x, y)}. Requires
     * matching rectangles.
     *
     * <p>Implementation: re-sample the pointwise product on a Chebyshev
     * grid, then run ACA on it. The product of a rank-{@code r_1} and a
     * rank-{@code r_2} approximation has separation rank at most
     * {@code r_1 * r_2}, so the result's rank can grow — ACA picks the
     * smallest rank that resolves to the requested tolerance. Sample
     * evaluation reuses each factor's {@code feval}, which is
     * {@code O(r_1 + r_2)} per point.
     */
    public Chebfun2 times(Chebfun2 other) {
        requireSameRect(other, "times");
        java.util.function.DoubleBinaryOperator productFn =
            (x, y) -> this.feval(x, y) * other.feval(x, y);
        return new Chebfun2(productFn, rect);
    }

    private Chebfun2 concatenate(Chebfun2 other, double sign) {
        int r1 = cols.length, r2 = other.cols.length;
        Chebfun[] c = new Chebfun[r1 + r2];
        Chebfun[] r = new Chebfun[r1 + r2];
        double[] p = new double[r1 + r2];
        System.arraycopy(cols, 0, c, 0, r1);
        System.arraycopy(rows, 0, r, 0, r1);
        System.arraycopy(pivotValues, 0, p, 0, r1);
        System.arraycopy(other.cols, 0, c, r1, r2);
        System.arraycopy(other.rows, 0, r, r1, r2);
        for (int i = 0; i < r2; i++) p[r1 + i] = other.pivotValues[i] / sign;
        return new Chebfun2(c, r, p, rect);
    }

    private void requireSameRect(Chebfun2 other, String op) {
        if (!rect.equalsRectangle(other.rect)) {
            throw new IllegalArgumentException(
                op + " requires matching rectangles: " + rect + " vs " + other.rect);
        }
    }
}
