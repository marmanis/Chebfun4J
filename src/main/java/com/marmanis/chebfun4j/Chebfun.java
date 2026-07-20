package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ChebyshevPoints;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 * A real function on a real interval {@code [a, b]}, represented as one or
 * more smooth {@link Fun} pieces glued at ascending <em>breakpoints</em>
 * {@code b_0 = a &lt; b_1 &lt; ... &lt; b_m = b}. Each piece {@code p_k}
 * lives on its own reference interval {@code [-1, 1]} and represents
 * {@code f} on the sub-interval {@code [b_k, b_{k+1}]} via the linear map
 * of {@link Domain}.
 *
 * <p>The smooth case is exactly {@code pieces.length == 1} and matches
 * iteration-1's single-piece Chebfun. Piecewise construction handles
 * functions with kinks or jumps (e.g. {@code |x|} at 0, {@code sign(x)})
 * via the <em>splitting-on</em> adaptive constructor: attempt a single
 * smooth piece; if the maximum grid length is reached without meeting the
 * happiness tolerance, bisect and recurse on each half. This mirrors
 * MATLAB chebfun's {@code chebfunpref('splitting', true)} default.
 *
 * <p>Same-type-only arithmetic: two Chebfuns whose pieces are all
 * {@link Chebtech} can be added, subtracted, multiplied; mixing bases
 * (e.g. Chebtech + a future Trigtech piece) is rejected with a clear
 * error until a common-basis coercion is added. Under the hood, differing
 * breakpoints are merged and each summand is restricted to each common
 * sub-interval before the piecewise op runs.
 */
public final class Chebfun {
    /** Relative tolerance for the resolution check ("machine precision"). */
    public static final double DEFAULT_TOL = 1e-14;

    /** Largest grid length the adaptive constructor will try on a single piece. */
    public static final int MAX_LENGTH = 65537; // 2^16 + 1

    /**
     * Cap on splitting-on recursion depth. Chosen so that the smallest
     * sub-interval before we stop bisecting is roughly {@code (b - a) / 2^20},
     * i.e. six-digit relative resolution of the breakpoint location — more
     * than adequate for locating a kink to within noise.
     */
    public static final int MAX_SPLIT_DEPTH = 20;

    /** Minimum sub-interval width below which splitting stops (relative). */
    public static final double MIN_SPLIT_WIDTH_REL = 1e-12;

    /**
     * Full grid sequence — used when we're committed to a single piece
     * (splitting has stopped or a piece is already known to be smooth).
     */
    private static final int[] GRID_LENGTHS = {
        17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385, 32769, 65537
    };

    /**
     * Shortened grid sequence used while splitting is still an option:
     * if we can't resolve a piece by length 129, we bisect rather than
     * chasing the tail all the way to 65537 (which is nearly free on a
     * smooth function but ruinously expensive multiplied by every kink-
     * containing ancestor interval on the recursion tree).
     */
    private static final int[] SPLIT_GRID_LENGTHS = {17, 33, 65, 129};

    private final Fun[] pieces;
    private final double[] breakpoints;
    private final Domain domain;

    /** Package-private: build directly from validated pieces + breakpoints. */
    Chebfun(Fun[] pieces, double[] breakpoints) {
        if (pieces.length + 1 != breakpoints.length) {
            throw new IllegalArgumentException(
                "breakpoints length must be pieces.length + 1, got " +
                pieces.length + " pieces and " + breakpoints.length + " breakpoints");
        }
        for (int i = 1; i < breakpoints.length; i++) {
            if (!(breakpoints[i - 1] < breakpoints[i])) {
                throw new IllegalArgumentException(
                    "breakpoints must be strictly ascending, got " + Arrays.toString(breakpoints));
            }
        }
        this.pieces = pieces;
        this.breakpoints = breakpoints;
        this.domain = new Domain(breakpoints[0], breakpoints[breakpoints.length - 1]);
    }

    /** Package-private: smooth single-piece constructor. */
    Chebfun(Fun tech, Domain domain) {
        this(new Fun[]{tech}, new double[]{domain.a(), domain.b()});
    }

    /**
     * Adaptively construct a chebfun of {@code f} on {@code domain} to
     * approximately {@link #DEFAULT_TOL} relative precision, with
     * splitting-on for functions that don't resolve as a single smooth
     * piece.
     */
    public Chebfun(DoubleUnaryOperator f, Domain domain) {
        this(f, domain, DEFAULT_TOL);
    }

    public Chebfun(DoubleUnaryOperator f, Domain domain, double tol) {
        // Two-phase approach: first try to resolve as a single smooth piece
        // with the full grid sequence (this is the common case — smooth
        // functions resolve at machine precision). Only fall back to
        // splitting if the smooth attempt fails at max grid, and then
        // stay on the short grid throughout the recursion — a leaf that
        // can't be resolved in 129 samples on a 2e-6-wide sub-interval
        // near a kink isn't going to be resolved at 65537 either, and
        // trying wastes O(n^2) work on later operations that touch it.
        Result whole = tryOnePiece(f, domain.a(), domain.b(), tol, GRID_LENGTHS);
        if (whole.happy) {
            this.pieces = new Fun[]{whole.tech.simplify(tol)};
            this.breakpoints = new double[]{domain.a(), domain.b()};
            this.domain = domain;
            return;
        }
        List<Fun> outPieces = new ArrayList<>();
        List<Double> outBreaks = new ArrayList<>();
        outBreaks.add(domain.a());
        splitAdaptive(f, domain.a(), domain.b(), tol, 0, outPieces, outBreaks);
        this.pieces = outPieces.toArray(new Fun[0]);
        this.breakpoints = outBreaks.stream().mapToDouble(Double::doubleValue).toArray();
        this.domain = domain;
    }

    /** The constant chebfun equal to {@code c} on {@code domain}. */
    public static Chebfun constant(double c, Domain domain) {
        return new Chebfun(Chebtech.constant(c), domain);
    }

    /**
     * Data-driven Chebfun: interpolate the sample points {@code (x_i, y_i)}
     * with a natural cubic spline (see
     * {@link com.marmanis.chebfun4j.util.CubicSpline}) and adaptively fit
     * the result on {@code domain} as a Chebfun.
     *
     * <p>Requires {@code x} strictly ascending and covering {@code domain}
     * (endpoint samples aren't required exactly at {@code a}/{@code b} —
     * the spline clamps outside its knot range, but for the fit to be
     * meaningful the knots should span the domain interior). Nice fit for
     * empirical data or samples generated by some other numerical process.
     */
    public static Chebfun fromSamples(double[] x, double[] y, Domain domain) {
        com.marmanis.chebfun4j.util.CubicSpline spline =
            com.marmanis.chebfun4j.util.CubicSpline.of(x, y);
        return new Chebfun(spline::eval, domain);
    }

    public Domain domain() {
        return domain;
    }

    /** Number of smooth pieces. Smooth-single-piece case returns {@code 1}. */
    public int numPieces() {
        return pieces.length;
    }

    /** Ascending breakpoints; length {@code numPieces() + 1}. */
    public double[] breakpoints() {
        return breakpoints.clone();
    }

    /** All smooth pieces. */
    public Fun[] pieces() {
        return pieces.clone();
    }

    /**
     * The single smooth piece if {@code numPieces() == 1}, else throws.
     * Convenience accessor for the smooth case.
     */
    public Fun tech() {
        if (pieces.length != 1) {
            throw new IllegalStateException(
                "tech() requires a single smooth piece, got " + pieces.length);
        }
        return pieces[0];
    }

    /**
     * Sum of piece lengths — total coefficient count across the whole
     * representation. For a smooth chebfun, this matches the old
     * {@code length()}.
     */
    public int length() {
        int total = 0;
        for (Fun p : pieces) total += p.length();
        return total;
    }

    /**
     * Evaluate {@code f(x)} for {@code x} in {@code [a, b]}. On a breakpoint
     * we return the value from the right-hand piece — consistent with
     * Chebfun's right-continuous convention.
     */
    public double feval(double x) {
        int k = locatePiece(x);
        double left = breakpoints[k];
        double right = breakpoints[k + 1];
        double y = (2 * x - (left + right)) / (right - left);
        return pieces[k].eval(clamp(y));
    }

    /** Add {@code other}; both must share the domain endpoints. */
    public Chebfun plus(Chebfun other) {
        requireSameDomain(other, "plus");
        return combine(other, BinaryOp.PLUS);
    }

    /** Subtract {@code other}; both must share the domain endpoints. */
    public Chebfun minus(Chebfun other) {
        requireSameDomain(other, "minus");
        return combine(other, BinaryOp.MINUS);
    }

    /** Pointwise product; both must share the domain endpoints. */
    public Chebfun times(Chebfun other) {
        requireSameDomain(other, "times");
        return combine(other, BinaryOp.TIMES);
    }

    private enum BinaryOp { PLUS, MINUS, TIMES }

    /**
     * Merge breakpoints of {@code this} and {@code other}, restrict each to
     * the common sub-intervals, apply the piecewise op, simplify.
     */
    private Chebfun combine(Chebfun other, BinaryOp op) {
        double[] merged = mergeBreakpoints(this.breakpoints, other.breakpoints);
        Fun[] outPieces = new Fun[merged.length - 1];
        for (int k = 0; k < outPieces.length; k++) {
            double lo = merged[k];
            double hi = merged[k + 1];
            Fun aP = restrictToInterval(this, lo, hi);
            Fun bP = restrictToInterval(other, lo, hi);
            outPieces[k] = binary(aP, bP, op).simplify(DEFAULT_TOL);
        }
        return new Chebfun(outPieces, merged);
    }

    private static Fun binary(Fun a, Fun b, BinaryOp op) {
        if (a instanceof Chebtech ac && b instanceof Chebtech bc) {
            return switch (op) {
                case PLUS  -> ac.plus(bc);
                case MINUS -> ac.minus(bc);
                case TIMES -> ac.times(bc);
            };
        }
        throw new UnsupportedOperationException(
            "Cross-basis piecewise arithmetic not yet supported: " +
            a.getClass().getSimpleName() + " " + op + " " + b.getClass().getSimpleName());
    }

    /** Unary negation. */
    public Chebfun negate() {
        Fun[] out = new Fun[pieces.length];
        for (int k = 0; k < pieces.length; k++) out[k] = pieces[k].negate();
        return new Chebfun(out, breakpoints.clone());
    }

    /** Add a scalar. */
    public Chebfun plus(double s) {
        Chebfun c = Chebfun.constant(s, domain);
        return this.plus(c);
    }

    /** Subtract a scalar. */
    public Chebfun minus(double s) {
        return this.plus(-s);
    }

    /** Scalar multiply. */
    public Chebfun times(double s) {
        Fun[] out = new Fun[pieces.length];
        for (int k = 0; k < pieces.length; k++) out[k] = pieces[k].times(s);
        return new Chebfun(out, breakpoints.clone());
    }

    /**
     * Pointwise division {@code f / other}. Requires the same domain and
     * that {@code other} is bounded away from zero — if it has roots inside
     * the domain the result is unbounded and the adaptive constructor will
     * refuse to resolve it.
     */
    public Chebfun divide(Chebfun other) {
        requireSameDomain(other, "divide");
        return new Chebfun(x -> this.feval(x) / other.feval(x), domain);
    }

    /**
     * Integer power {@code f^n}, computed via fast exponentiation over
     * {@link #times(Chebfun)}. For {@code n == 0} returns the constant 1;
     * for {@code n < 0} returns {@code 1 / f^{-n}} (and inherits the
     * "divide by potentially zero" caveat of {@link #divide}).
     */
    public Chebfun pow(int n) {
        if (n == 0) return Chebfun.constant(1.0, domain);
        if (n < 0) return Chebfun.constant(1.0, domain).divide(this.pow(-n));
        Chebfun result = null;
        Chebfun base = this;
        int e = n;
        while (e > 0) {
            if ((e & 1) == 1) result = (result == null) ? base : result.times(base);
            e >>= 1;
            if (e > 0) base = base.times(base);
        }
        return result;
    }

    /**
     * Pointwise real power {@code f^p}. Uses the adaptive Chebfun
     * constructor to resolve {@code x -> pow(f(x), p)}; caller is
     * responsible for {@code f} being non-negative when {@code p} is
     * fractional.
     */
    public Chebfun pow(double p) {
        return new Chebfun(x -> Math.pow(this.feval(x), p), domain);
    }

    /**
     * Pointwise absolute value {@code |f|}. Where {@code f} has zeros in
     * the interior of the domain, {@code |f|} has a kink and the adaptive
     * splitting-on constructor bisects to represent it exactly.
     */
    public Chebfun abs() {
        return new Chebfun(x -> Math.abs(this.feval(x)), domain);
    }

    /**
     * Composition {@code f ∘ g}, i.e. {@code (f ∘ g)(x) = f(g(x))}. The
     * result lives on {@code g}'s domain. For the composition to be
     * mathematically meaningful the range of {@code g} should lie in
     * {@code f}'s domain — if it doesn't, {@code f.feval} clamps to the
     * nearest endpoint (numerically extrapolating rather than throwing).
     */
    public Chebfun compose(Chebfun g) {
        return new Chebfun(x -> this.feval(g.feval(x)), g.domain());
    }

    /**
     * Parallel arrays of {@code n} uniformly-spaced sample points and the
     * function values at them: {@code x[i] = a + i * (b - a) / (n - 1)},
     * {@code y[i] = feval(x[i])}. Convenient handoff to any plotting or
     * numerical library — no additional dependency in chebfun4j itself.
     *
     * <p>Uniform (not Chebyshev) spacing so downstream plotters and
     * consumers get the shape they usually expect. For n=1, returns a
     * single-point sample at the domain's midpoint.
     */
    public Samples samples(int n) {
        if (n < 1) throw new IllegalArgumentException("samples requires n >= 1, got " + n);
        double a = domain.a();
        double b = domain.b();
        double[] x = new double[n];
        double[] y = new double[n];
        if (n == 1) {
            x[0] = 0.5 * (a + b);
            y[0] = feval(x[0]);
            return new Samples(x, y);
        }
        double dx = (b - a) / (n - 1);
        for (int i = 0; i < n; i++) {
            x[i] = a + i * dx;
            y[i] = feval(x[i]);
        }
        return new Samples(x, y);
    }

    /** Result of {@link #samples(int)}: parallel arrays of x and y=f(x). */
    public record Samples(double[] x, double[] y) {}

    /**
     * Definite integral {@code integral_a^b f(x) dx}. Sum of per-piece
     * integrals, each rescaled by the piece width.
     */
    public double sum() {
        double total = 0.0;
        for (int k = 0; k < pieces.length; k++) {
            double width = breakpoints[k + 1] - breakpoints[k];
            total += 0.5 * width * pieces[k].sum();
        }
        return total;
    }

    /**
     * Definite integral over the sub-interval {@code [a, b]} of the domain:
     * {@code F(b) - F(a)}, where {@code F} is the antiderivative from
     * {@link #cumsum()}. Both endpoints must lie in the domain (order
     * irrelevant — {@code sum(a, b) == -sum(b, a)}). O(n²) once for
     * {@code cumsum}, then two {@code feval}s.
     */
    public double sum(double a, double b) {
        double lo = Math.min(a, b);
        double hi = Math.max(a, b);
        if (lo < domain.a() - 1e-14 || hi > domain.b() + 1e-14) {
            throw new IllegalArgumentException(
                "sum(a, b) requires both endpoints in " + domain + ", got [" + a + ", " + b + "]");
        }
        Chebfun F = this.cumsum();
        double signed = F.feval(hi) - F.feval(lo);
        return (b >= a) ? signed : -signed;
    }

    /**
     * Derivative {@code f'}. Piecewise: each piece is differentiated on its
     * reference interval and multiplied by {@code 2 / (b_{k+1} - b_k)}.
     * Breakpoints are preserved (the derivative is generally discontinuous
     * at a kink; the piecewise chebfun captures that).
     */
    public Chebfun diff() {
        Fun[] out = new Fun[pieces.length];
        for (int k = 0; k < pieces.length; k++) {
            double jac = 2.0 / (breakpoints[k + 1] - breakpoints[k]);
            out[k] = pieces[k].diff().times(jac).simplify(DEFAULT_TOL);
        }
        return new Chebfun(out, breakpoints.clone());
    }

    /**
     * Indefinite integral {@code F} such that {@code F(a) = 0}. Piecewise
     * with the constant of integration threaded through the breakpoints
     * so that {@code F} is continuous even when {@code f} has jumps
     * (an indefinite integral is always continuous, jumps of {@code f}
     * only produce kinks in {@code F}).
     */
    public Chebfun cumsum() {
        Fun[] out = new Fun[pieces.length];
        double running = 0.0;
        for (int k = 0; k < pieces.length; k++) {
            double halfWidth = 0.5 * (breakpoints[k + 1] - breakpoints[k]);
            Fun local = pieces[k].cumsum().times(halfWidth);
            if (running != 0.0) {
                Fun shift = Chebtech.constant(running);
                if (local instanceof Chebtech lc) local = lc.plus((Chebtech) shift);
                else throw new UnsupportedOperationException(
                    "cumsum on non-Chebtech piece not supported yet");
            }
            out[k] = local.simplify(DEFAULT_TOL);
            // Update the running constant using the piece's right-endpoint value.
            running = out[k].eval(1.0);
        }
        return new Chebfun(out, breakpoints.clone());
    }

    /**
     * Real roots of {@code f} in {@code [a, b]}, sorted ascending. Collects
     * per-piece roots (mapped back to global coordinates) and dedup's at
     * breakpoints where an interior root of one piece coincides with an
     * endpoint root of the neighbour.
     */
    public double[] roots() {
        List<Double> found = new ArrayList<>();
        for (int k = 0; k < pieces.length; k++) {
            double lo = breakpoints[k];
            double hi = breakpoints[k + 1];
            double mid = 0.5 * (lo + hi);
            double halfWidth = 0.5 * (hi - lo);
            for (double y : pieces[k].rootsOnRef()) {
                found.add(mid + halfWidth * y);
            }
        }
        java.util.Collections.sort(found);
        double eps = 1e-11 * Math.max(1.0, domain.length());
        List<Double> dedup = new ArrayList<>(found.size());
        double last = Double.NEGATIVE_INFINITY;
        for (double r : found) {
            if (r - last > eps) {
                dedup.add(r);
                last = r;
            }
        }
        double[] out = new double[dedup.size()];
        for (int i = 0; i < out.length; i++) out[i] = dedup.get(i);
        return out;
    }

    /**
     * Both the extremum's value and its location.
     */
    public record Extremum(double value, double location) {}

    /**
     * Global minimum on {@code [a, b]}. Critical points of {@code f'}
     * within each piece plus all breakpoints (which are candidates because
     * {@code f} may be non-differentiable there).
     */
    public Extremum min() {
        return findExtremum(true);
    }

    /** Global maximum on {@code [a, b]}. See {@link #min()}. */
    public Extremum max() {
        return findExtremum(false);
    }

    private Extremum findExtremum(boolean wantMin) {
        double bestX = breakpoints[0];
        double bestV = this.feval(bestX);
        for (int i = 1; i < breakpoints.length; i++) {
            double x = breakpoints[i];
            double v = this.feval(x);
            if ((wantMin && v < bestV) || (!wantMin && v > bestV)) { bestX = x; bestV = v; }
        }
        Chebfun df = this.diff();
        double[] critical = df.roots();
        for (double x : critical) {
            double v = this.feval(x);
            if ((wantMin && v < bestV) || (!wantMin && v > bestV)) { bestX = x; bestV = v; }
        }
        return new Extremum(bestV, bestX);
    }

    /** Uniform ({@code L^infinity}) norm — the maximum absolute value on {@code [a, b]}. */
    public double normInf() {
        double lo = Math.abs(this.min().value());
        double hi = Math.abs(this.max().value());
        return Math.max(lo, hi);
    }

    /** {@code L^2} norm: {@code sqrt(integral_a^b f^2)}. */
    public double norm2() {
        Chebfun f2 = this.times(this);
        return Math.sqrt(Math.max(0.0, f2.sum()));
    }

    /**
     * {@code L^1} norm: {@code integral_a^b |f|}. Split at zeros; add the
     * absolute value of the antiderivative on each sign-constant piece.
     */
    public double norm1() {
        double[] r = this.roots();
        Chebfun F = this.cumsum();
        double total = 0.0;
        double Fleft = 0.0;
        for (double x : r) {
            double Fx = F.feval(x);
            total += Math.abs(Fx - Fleft);
            Fleft = Fx;
        }
        double Fb = F.feval(domain.b());
        total += Math.abs(Fb - Fleft);
        return total;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    private static double clamp(double y) {
        if (y < -1.0) return -1.0;
        if (y >  1.0) return  1.0;
        return y;
    }

    private int locatePiece(double x) {
        // Binary search for the interval containing x (right-open until b).
        if (x <= breakpoints[0]) return 0;
        if (x >= breakpoints[breakpoints.length - 1]) return pieces.length - 1;
        int lo = 0, hi = pieces.length;
        while (lo < hi - 1) {
            int mid = (lo + hi) >>> 1;
            if (x < breakpoints[mid]) hi = mid; else lo = mid;
        }
        return lo;
    }

    private static double[] mergeBreakpoints(double[] a, double[] b) {
        double lo = Math.min(a[0], b[0]);
        double hi = Math.max(a[a.length - 1], b[b.length - 1]);
        // Domain endpoints must actually agree — check performed by caller.
        // Sorted merge with dedup within a relative tolerance.
        double tol = 1e-13 * Math.max(1.0, hi - lo);
        List<Double> out = new ArrayList<>();
        int i = 0, j = 0;
        while (i < a.length || j < b.length) {
            double va = (i < a.length) ? a[i] : Double.POSITIVE_INFINITY;
            double vb = (j < b.length) ? b[j] : Double.POSITIVE_INFINITY;
            double pick;
            if (Math.abs(va - vb) <= tol) { pick = 0.5 * (va + vb); i++; j++; }
            else if (va < vb) { pick = va; i++; }
            else { pick = vb; j++; }
            if (out.isEmpty() || pick - out.get(out.size() - 1) > tol) out.add(pick);
        }
        double[] arr = new double[out.size()];
        for (int k = 0; k < arr.length; k++) arr[k] = out.get(k);
        return arr;
    }

    /**
     * Return the Fun that represents {@code f} restricted to the sub-
     * interval {@code [lo, hi]}, expressed on its own local reference
     * {@code [-1, 1]}. Sampling is done by feval, so the caller pays a
     * Clenshaw eval per grid point but doesn't need to touch the internal
     * piecewise structure. Length is chosen to match the widest overlapping
     * piece so we don't lose resolution.
     */
    private static Fun restrictToInterval(Chebfun f, double lo, double hi) {
        int n = 1;
        for (int k = 0; k < f.pieces.length; k++) {
            if (f.breakpoints[k] < hi && f.breakpoints[k + 1] > lo) {
                n = Math.max(n, f.pieces[k].length());
            }
        }
        // Ensure a valid 2^k + 1 length for the FFT-based transform.
        n = com.marmanis.chebfun4j.util.ChebyshevPoints.nextValidLength(Math.max(n, 3));
        double[] refGrid = ChebyshevPoints.secondKind(n);
        double mid = 0.5 * (lo + hi);
        double halfWidth = 0.5 * (hi - lo);
        double[] vals = new double[n];
        for (int i = 0; i < n; i++) {
            double x = mid + halfWidth * refGrid[i];
            vals[i] = f.feval(x);
        }
        return Chebtech.fromValues(vals);
    }

    private void requireSameDomain(Chebfun other, String op) {
        if (!domain.equalsDomain(other.domain)) {
            throw new IllegalArgumentException(
                op + " requires matching domains: " + domain + " vs " + other.domain);
        }
    }

    // ------------------------------------------------------------------
    // Adaptive splitting-on constructor
    // ------------------------------------------------------------------

    private static void splitAdaptive(DoubleUnaryOperator f,
                                       double a, double b,
                                       double tol,
                                       int depth,
                                       List<Fun> outPieces,
                                       List<Double> outBreaks) {
        boolean tinyWidth = (b - a) <= MIN_SPLIT_WIDTH_REL * Math.max(1.0, Math.abs(a) + Math.abs(b));
        Result r = tryOnePiece(f, a, b, tol, SPLIT_GRID_LENGTHS);
        if (!r.happy && depth < MAX_SPLIT_DEPTH && !tinyWidth) {
            double mid = 0.5 * (a + b);
            splitAdaptive(f, a, mid, tol, depth + 1, outPieces, outBreaks);
            splitAdaptive(f, mid, b, tol, depth + 1, outPieces, outBreaks);
            return;
        }
        // Accept whatever the short-grid probe gave us. A stubbornly
        // unhappy leaf on a tiny sub-interval keeps its 129-coefficient
        // approximation rather than being blown up to 65537 — the extra
        // coefficients would be noise, not signal, and every downstream
        // operation ({@link RootFinder#subdividedRoots} is the worst
        // offender) pays O(length^2) to touch them.
        outPieces.add(r.tech.simplify(tol));
        outBreaks.add(b);
    }

    private record Result(Chebtech tech, boolean happy) {}

    private static Result tryOnePiece(DoubleUnaryOperator f, double a, double b,
                                      double tol, int[] gridLengths) {
        Chebtech last = null;
        double mid = 0.5 * (a + b);
        double halfWidth = 0.5 * (b - a);
        for (int n : gridLengths) {
            double[] refGrid = ChebyshevPoints.secondKind(n);
            double[] values = new double[n];
            for (int i = 0; i < n; i++) {
                double x = mid + halfWidth * refGrid[i];
                values[i] = f.applyAsDouble(x);
            }
            Chebtech candidate = Chebtech.fromValues(values);
            last = candidate;
            if (isResolved(candidate, values, tol)) {
                return new Result(candidate, true);
            }
        }
        return new Result(last, false);
    }

    private static boolean isResolved(Chebtech candidate, double[] values, double tol) {
        double vscale = 0.0;
        for (double v : values) vscale = Math.max(vscale, Math.abs(v));
        if (vscale == 0.0) vscale = 1.0;
        double cutoff = tol * vscale;
        double[] c = candidate.coeffs();
        int n = c.length;
        int start = n / 2;
        double tailMax = 0.0;
        for (int k = start; k < n; k++) tailMax = Math.max(tailMax, Math.abs(c[k]));
        return tailMax <= cutoff;
    }
}
