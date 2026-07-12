package com.marmanis.chebfun4j;

import java.util.function.DoubleUnaryOperator;

/**
 * A smooth <em>periodic</em> real function on a real interval {@code [a, b]}
 * (implied period {@code b - a}), represented as a single {@link Trigtech}
 * on {@code [-1, 1]} composed with {@link Domain}'s linear map. This is the
 * periodic counterpart of {@link Chebfun} restricted to a single smooth
 * piece — chebfun4j does not (yet) piecewise-glue trigtechs.
 *
 * <p>Constructor is adaptive: sample on progressively finer equispaced
 * (periodic) grids of size {@code 32, 64, 128, ..., 65536} until the tail
 * Fourier coefficients drop below {@code tol * vscale}. Grid lengths are
 * powers of two so the underlying FFT stays on the fast path.
 */
public final class Trigfun {
    public static final double DEFAULT_TOL = 1e-14;
    public static final int MAX_LENGTH = 1 << 16; // 65536

    private static final int[] GRID_LENGTHS = {
        32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    };

    private final Trigtech tech;
    private final Domain domain;

    Trigfun(Trigtech tech, Domain domain) {
        this.tech = tech;
        this.domain = domain;
    }

    /**
     * Constant trigfun equal to {@code c} on {@code domain}.
     */
    public static Trigfun constant(double c, Domain domain) {
        return new Trigfun(Trigtech.constant(c), domain);
    }

    /**
     * Adaptively construct a trigfun of the periodic function {@code f} on
     * {@code domain} to approximately {@link #DEFAULT_TOL} relative
     * precision.
     */
    public Trigfun(DoubleUnaryOperator f, Domain domain) {
        this(f, domain, DEFAULT_TOL);
    }

    public Trigfun(DoubleUnaryOperator f, Domain domain, double tol) {
        this.domain = domain;
        this.tech = adaptivelyConstruct(f, domain, tol);
    }

    public Domain domain() { return domain; }
    public Trigtech tech() { return tech; }
    public int length() { return tech.length(); }

    /** Evaluate {@code f(x)} for {@code x} in {@code [a, b]}. */
    public double feval(double x) {
        return tech.eval(domain.toRef(x));
    }

    /** Add {@code other}; both must share the domain endpoints. */
    public Trigfun plus(Trigfun other) {
        requireSameDomain(other, "plus");
        return finish(tech.plus(other.tech));
    }

    public Trigfun minus(Trigfun other) {
        requireSameDomain(other, "minus");
        return finish(tech.minus(other.tech));
    }

    public Trigfun times(Trigfun other) {
        requireSameDomain(other, "times");
        return finish(tech.times(other.tech));
    }

    public Trigfun negate() {
        return finish(tech.negate());
    }

    public Trigfun times(double s) {
        return finish(tech.times(s));
    }

    public Trigfun plus(double s) {
        Trigtech shifted = tech.plus(Trigtech.constant(s));
        return finish(shifted);
    }

    public Trigfun minus(double s) {
        return plus(-s);
    }

    /**
     * Definite integral {@code integral_a^b f}. Since the reference-integral
     * {@link Trigtech#sum()} equals {@code 2 a_0} (the mean over one period,
     * times the period-2 width), the domain scale factor is
     * {@code (b - a) / 2}.
     */
    public double sum() {
        return 0.5 * domain.length() * tech.sum();
    }

    /**
     * Derivative {@code f'}. Reference derivative {@code d/dy}, scaled by
     * {@code 2 / (b - a)} for the chain rule.
     */
    public Trigfun diff() {
        double jac = 2.0 / domain.length();
        return finish(tech.diff().times(jac));
    }

    private Trigfun finish(Trigtech t) {
        return new Trigfun(t.simplify(DEFAULT_TOL), domain);
    }

    private void requireSameDomain(Trigfun other, String op) {
        if (!domain.equalsDomain(other.domain)) {
            throw new IllegalArgumentException(
                op + " requires matching domains: " + domain + " vs " + other.domain);
        }
    }

    private static Trigtech adaptivelyConstruct(DoubleUnaryOperator f, Domain domain, double tol) {
        Trigtech last = null;
        for (int n : GRID_LENGTHS) {
            double[] values = new double[n];
            // Equispaced periodic grid: y_j = -1 + 2 j / N (does NOT include right endpoint).
            double dy = 2.0 / n;
            for (int j = 0; j < n; j++) {
                double y = -1.0 + j * dy;
                values[j] = f.applyAsDouble(domain.fromRef(y));
            }
            Trigtech candidate = Trigtech.fromValues(values);
            last = candidate;
            if (isResolved(candidate, values, tol)) {
                return candidate.simplify(tol);
            }
        }
        return last;
    }

    private static boolean isResolved(Trigtech candidate, double[] values, double tol) {
        double vscale = 0.0;
        for (double v : values) vscale = Math.max(vscale, Math.abs(v));
        if (vscale == 0.0) vscale = 1.0;
        double cutoff = tol * vscale;
        double[] a = candidate.cosCoeffs();
        double[] b = candidate.sinCoeffs();
        int m = a.length;
        int start = m / 2;
        double tailMax = 0.0;
        for (int k = start; k < m; k++) {
            tailMax = Math.max(tailMax, Math.abs(a[k]));
            tailMax = Math.max(tailMax, Math.abs(b[k]));
        }
        return tailMax <= cutoff;
    }
}
