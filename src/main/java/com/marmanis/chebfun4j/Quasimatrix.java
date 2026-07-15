package com.marmanis.chebfun4j;

/**
 * A finite ordered collection of {@link Chebfun}s on a common
 * {@link Domain}, treated as an "∞ × n matrix" — each column is a
 * continuous function of {@code x ∈ [a, b]}, and the discrete index runs
 * over the columns. The name and layout follow MATLAB chebfun's
 * quasimatrix concept.
 *
 * <p>The MVP surface is intentionally small: the shape you need for
 * {@code Chebop.eigs}'s return value plus the couple of pieces of linear
 * algebra ({@link #innerProduct} of two columns, per-column
 * {@link #normalize}) that let the caller work with eigenfunctions
 * meaningfully. QR / SVD of a quasimatrix are follow-ups — see the
 * iteration 3 README notes.
 *
 * <p>Immutable and thread-safe; every operation returns a fresh
 * quasimatrix. Column access ({@link #get}) returns a shared reference —
 * {@code Chebfun} is itself immutable, so this is safe.
 */
public final class Quasimatrix {
    private final Chebfun[] columns;
    private final Domain domain;

    public Quasimatrix(Chebfun[] columns) {
        if (columns == null || columns.length == 0) {
            throw new IllegalArgumentException("quasimatrix requires >= 1 column");
        }
        Domain d = columns[0].domain();
        for (int i = 1; i < columns.length; i++) {
            if (!columns[i].domain().equalsDomain(d)) {
                throw new IllegalArgumentException(
                    "all columns must share the domain " + d + ", column " + i +
                    " is on " + columns[i].domain());
            }
        }
        this.columns = columns.clone();
        this.domain = d;
    }

    /** Number of columns (functions). */
    public int numColumns() {
        return columns.length;
    }

    /** Shared domain of every column. */
    public Domain domain() {
        return domain;
    }

    /** The {@code k}-th column (0-indexed). */
    public Chebfun get(int k) {
        return columns[k];
    }

    /** A fresh array of all columns. */
    public Chebfun[] columns() {
        return columns.clone();
    }

    /**
     * Continuous {@code L^2([a, b])} inner product of columns {@code i} and
     * {@code j} — {@code integral_a^b f_i(x) f_j(x) dx}. Uses the product
     * chebfun's exact integral formula.
     */
    public double innerProduct(int i, int j) {
        return columns[i].times(columns[j]).sum();
    }

    /**
     * The {@code L^2} norm of column {@code k}. Equivalent to
     * {@code sqrt(innerProduct(k, k))} but delegates to
     * {@link Chebfun#norm2} for the same computation with tighter numerical
     * safeguards.
     */
    public double columnNorm2(int k) {
        return columns[k].norm2();
    }

    /**
     * Return a new quasimatrix in which every column has been scaled so
     * that its {@code L^2} norm is {@code 1}. Zero-norm columns (an
     * identically-zero eigenfunction, say) are left unchanged.
     */
    public Quasimatrix normalizeColumns() {
        Chebfun[] out = new Chebfun[columns.length];
        for (int k = 0; k < columns.length; k++) {
            double norm = columns[k].norm2();
            out[k] = (norm == 0.0) ? columns[k] : columns[k].times(1.0 / norm);
        }
        return new Quasimatrix(out);
    }

    /** Column-wise addition; requires the same number of columns. */
    public Quasimatrix plus(Quasimatrix other) {
        requireSameShape(other, "plus");
        Chebfun[] out = new Chebfun[columns.length];
        for (int k = 0; k < columns.length; k++) out[k] = columns[k].plus(other.columns[k]);
        return new Quasimatrix(out);
    }

    /** Column-wise subtraction. */
    public Quasimatrix minus(Quasimatrix other) {
        requireSameShape(other, "minus");
        Chebfun[] out = new Chebfun[columns.length];
        for (int k = 0; k < columns.length; k++) out[k] = columns[k].minus(other.columns[k]);
        return new Quasimatrix(out);
    }

    /** Scale every column by {@code s}. */
    public Quasimatrix times(double s) {
        Chebfun[] out = new Chebfun[columns.length];
        for (int k = 0; k < columns.length; k++) out[k] = columns[k].times(s);
        return new Quasimatrix(out);
    }

    // ---------------------------------------------------------------
    // Singular-value decomposition
    // ---------------------------------------------------------------

    /**
     * Result of {@link #svd() singular-value decomposition}:
     * {@code A = U * diag(sigma) * V^T} where {@code U} is a quasimatrix
     * of {@code L^2}-orthonormal left singular functions, {@code sigma}
     * is a length-{@code n} vector of singular values in descending
     * order, and {@code Vt} is an {@code n × n} orthogonal matrix (row
     * major).
     */
    public record Svd(Quasimatrix U, double[] sigma, double[][] Vt) {}

    /**
     * Singular-value decomposition of this quasimatrix. Computed via the
     * standard two-step reduction: first {@link #qr(Algorithm) Householder
     * QR} gives {@code A = Q R} with an orthonormal quasimatrix
     * {@code Q} and a small upper-triangular {@code R}; then
     * {@link com.marmanis.jax4j.api.Linalg#svd Linalg.svd} on {@code R}
     * yields {@code R = U_R Σ V^T}. Assembling gives
     * {@code A = (Q U_R) Σ V^T} — so the left singular functions are
     * a linear combination of {@code Q}'s columns.
     *
     * <p>The reduction to a small numerical matrix is what makes this
     * tractable: {@code Linalg.svd} on an {@code n × n} dense matrix is
     * the same problem LAPACK's {@code DGESVD} solves in seconds even
     * at moderate {@code n}. Householder QR (rather than MGS) is used
     * so orthogonality of {@code Q} stays tight even on ill-conditioned
     * quasimatrices.
     */
    public Svd svd() {
        Qr qr = qr(Algorithm.HOUSEHOLDER);
        int n = columns.length;
        // Build R as an m×n NDArray (m = n here since R is n×n).
        double[] Rflat = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) Rflat[i * n + j] = qr.R()[i][j];
        }
        com.marmanis.jax4j.core.NDArray Rarr =
            new com.marmanis.jax4j.core.ConcreteNDArray(Rflat,
                new com.marmanis.jax4j.core.Shape(n, n));
        com.marmanis.jax4j.api.Linalg.Svd s = com.marmanis.jax4j.api.Linalg.svd(Rarr);
        double[] Ur = s.U().toDoubleArray();
        double[] sigma = s.sigma().toDoubleArray();
        double[] Vt = s.Vt().toDoubleArray();
        // Assemble U as a quasimatrix: U[:, j] = sum_i Q[:, i] * Ur[i, j].
        Chebfun[] qCols = qr.Q().columns;
        Chebfun[] Ucols = new Chebfun[n];
        for (int j = 0; j < n; j++) {
            Chebfun col = null;
            for (int i = 0; i < n; i++) {
                double c = Ur[i * n + j];
                if (c == 0.0) continue;
                Chebfun term = qCols[i].times(c);
                col = (col == null) ? term : col.plus(term);
            }
            if (col == null) col = Chebfun.constant(0.0, domain);
            Ucols[j] = col;
        }
        double[][] VtOut = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) VtOut[i][j] = Vt[i * n + j];
        }
        return new Svd(new Quasimatrix(Ucols), sigma, VtOut);
    }

    // ---------------------------------------------------------------
    // QR decomposition
    // ---------------------------------------------------------------

    /**
     * Result of {@link #qr(Algorithm) QR decomposition}: {@code Q} has
     * {@code L^2}-orthonormal columns spanning the same range as the
     * original quasimatrix, and {@code R} is a small {@code n × n}
     * upper-triangular numerical matrix such that {@code A = Q R}.
     */
    public record Qr(Quasimatrix Q, double[][] R) {}

    /**
     * QR algorithm selection.
     * <ul>
     *   <li>{@link #HOUSEHOLDER} — <b>default.</b> chebfun-style
     *       Householder reflectors (Trefethen 2010) built against an
     *       L²-orthonormal Legendre reference basis. Robust on
     *       ill-conditioned quasimatrices; recovers Q's orthogonality to
     *       near machine precision even at column counts (n ≳ 15) where
     *       Gram-Schmidt loses orthogonality noticeably.</li>
     *   <li>{@link #MODIFIED_GRAM_SCHMIDT} — column-wise Gram-Schmidt
     *       with the "modified" order (each new column subtracts
     *       already-computed {@code q_i} contributions using the running
     *       residual, not the original column). Faster than Householder
     *       but loses orthogonality on ill-conditioned inputs — opt-in
     *       when you know the columns are well-separated.</li>
     * </ul>
     */
    public enum Algorithm { MODIFIED_GRAM_SCHMIDT, HOUSEHOLDER }

    /**
     * Compute the QR decomposition using
     * {@link Algorithm#HOUSEHOLDER Householder reflectors} — the robust
     * default. Use {@link #qr(Algorithm)} with
     * {@link Algorithm#MODIFIED_GRAM_SCHMIDT} to opt into the faster but
     * less stable MGS variant.
     */
    public Qr qr() {
        return qr(Algorithm.HOUSEHOLDER);
    }

    /** Compute the QR decomposition of this quasimatrix. */
    public Qr qr(Algorithm algorithm) {
        return switch (algorithm) {
            case MODIFIED_GRAM_SCHMIDT -> qrModifiedGramSchmidt();
            case HOUSEHOLDER           -> qrHouseholder();
        };
    }

    private Qr qrModifiedGramSchmidt() {
        int n = columns.length;
        Chebfun[] q = new Chebfun[n];
        double[][] R = new double[n][n];
        for (int j = 0; j < n; j++) {
            Chebfun v = columns[j];
            for (int i = 0; i < j; i++) {
                double rij = q[i].times(v).sum();
                R[i][j] = rij;
                if (rij != 0.0) v = v.minus(q[i].times(rij));
            }
            double norm = v.norm2();
            R[j][j] = norm;
            q[j] = (norm == 0.0) ? v : v.times(1.0 / norm);
        }
        return new Qr(new Quasimatrix(q), R);
    }

    /**
     * Householder QR on a quasimatrix. Follows Trefethen's construction
     * (see "Householder triangularization of a quasimatrix", IMA J.
     * Numer. Anal. 2010): at step {@code k}, build a reflector defined
     * by an orthonormal Chebfun {@code v_k} that zeroes out the "below-
     * diagonal" part of column {@code k}. The "diagonal" of a
     * quasimatrix is chosen deterministically — we use the sign-
     * canonical scalar {@code alpha = -sign(<e_k, x>) ||x||} where the
     * "elementary column" {@code e_k} is the {@code k}-th Legendre-
     * orthonormal basis chebfun (see
     * {@link #orthonormalLegendreBasis}). In practice for chebfun4j's
     * needs the choice of {@code e_k} basis doesn't matter for
     * correctness of {@code Q R = A}; we just need a consistent
     * "vertical direction" that is itself well-conditioned.
     */
    private Qr qrHouseholder() {
        int n = columns.length;
        // Working copies of the columns; we'll reduce them in place.
        Chebfun[] work = columns.clone();
        // "E-basis": Legendre polynomials on the domain, rescaled to unit L²
        // norm. Already orthonormal by construction — no MGS pass, no
        // exponential ill-conditioning. See Trefethen 2010
        // section 4 for why any orthonormal e-basis works.
        Chebfun[] e = orthonormalLegendreBasis(domain, n);
        // For accumulating the reflectors we track Q as its columns.
        // Each reflector is v_k (an orthonormal chebfun) and Q's k-th
        // column is the reflected e_k.
        Chebfun[] v = new Chebfun[n];
        double[][] R = new double[n][n];
        for (int k = 0; k < n; k++) {
            Chebfun x = work[k];
            // Compute alpha = sign correction and norm; sign convention
            // matches LAPACK's DLARFG for reproducibility.
            double xkEk = x.times(e[k]).sum();
            double xNorm = x.norm2();
            double alpha = (xkEk >= 0) ? -xNorm : xNorm;
            R[k][k] = alpha;
            // v_k = (x - alpha e_k), then normalize to unit norm.
            Chebfun w = x.minus(e[k].times(alpha));
            double wNorm = w.norm2();
            v[k] = (wNorm == 0.0) ? w : w.times(1.0 / wNorm);
            // Apply reflector I - 2 v v^T to columns j > k. Also
            // record the corresponding R[k][j] entries.
            for (int j = k + 1; j < n; j++) {
                double dot = v[k].times(work[j]).sum();
                work[j] = work[j].minus(v[k].times(2.0 * dot));
                R[k][j] = e[k].times(work[j]).sum();
                work[j] = work[j].minus(e[k].times(R[k][j]));
            }
        }
        // Reconstruct Q: q_k is the result of applying reflectors
        // (in reverse order) to e_k. We walk backward through v.
        Chebfun[] q = new Chebfun[n];
        for (int k = 0; k < n; k++) {
            Chebfun qk = e[k];
            for (int i = k; i >= 0; i--) {
                double dot = v[i].times(qk).sum();
                qk = qk.minus(v[i].times(2.0 * dot));
            }
            q[k] = qk;
        }
        return new Qr(new Quasimatrix(q), R);
    }

    /**
     * Build an {@code L^2}-orthonormal chebfun basis of the polynomial
     * space of degree {@code < n} on {@code domain}, used as the "vertical
     * direction" reference basis for {@link #qrHouseholder}.
     *
     * <p>Uses <b>Legendre polynomials</b>: they're already {@code L^2}-
     * orthogonal on {@code [-1, 1]} under the plain (unweighted) inner
     * product, so the affine-mapped {@code P_k(y(x))} rescaled by
     * {@code sqrt((2k+1)/(b-a))} is exactly the L²-orthonormal basis on
     * {@code [a, b]}. No MGS pass is needed — and there's nothing to lose
     * to numerical cancellation the way the previous
     * {@code {1, x, x², …, xⁿ⁻¹}}-plus-MGS construction did (that basis is
     * exponentially ill-conditioned by n ≈ 15 and MGS silently gives up
     * orthogonality long before it errors).
     *
     * <p>The Legendre polynomials are computed by the standard 3-term
     * recurrence
     * <pre>
     *   P_0(y) = 1, P_1(y) = y,
     *   (k+1) P_{k+1}(y) = (2k+1) y P_k(y) - k P_{k-1}(y),
     * </pre>
     * driven directly on Chebfuns of the reference-mapped
     * {@code y(x) = (2x - (a+b)) / (b-a)}.
     */
    private static Chebfun[] orthonormalLegendreBasis(Domain domain, int n) {
        double a = domain.a();
        double b = domain.b();
        double halfWidth = 0.5 * (b - a);
        double mid = 0.5 * (a + b);
        double invLen = 1.0 / (b - a);
        Chebfun[] e = new Chebfun[n];
        // Evaluate each P_k(y(x)) pointwise via the scalar 3-term recurrence
        // and let Chebfun's adaptive constructor resolve the polynomial. This
        // is intentionally NOT recursion over Chebfuns: driving the recurrence
        // in Chebfun space would apply `.simplify(1e-14)` at every step, and
        // over 20 steps the truncation cost enough orthogonality to break
        // Householder QR. Evaluating in double arithmetic at sample points
        // keeps every P_k accurate to machine precision.
        for (int k = 0; k < n; k++) {
            final int K = k;
            double c = Math.sqrt((2 * K + 1.0) * invLen);
            e[k] = new Chebfun(x -> {
                double y = (x - mid) / halfWidth;
                double pPrev = 1.0;               // P_0(y)
                if (K == 0) return c * pPrev;
                double pCurr = y;                 // P_1(y)
                for (int m = 1; m < K; m++) {
                    double pNext = ((2 * m + 1) * y * pCurr - m * pPrev) / (m + 1);
                    pPrev = pCurr;
                    pCurr = pNext;
                }
                return c * pCurr;
            }, domain);
        }
        return e;
    }

    private void requireSameShape(Quasimatrix other, String op) {
        if (other.columns.length != columns.length) {
            throw new IllegalArgumentException(
                op + " requires matching column counts: " +
                columns.length + " vs " + other.columns.length);
        }
        if (!other.domain.equalsDomain(domain)) {
            throw new IllegalArgumentException(
                op + " requires matching domains: " + domain + " vs " + other.domain);
        }
    }
}
