package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ChebTransform;
import com.marmanis.chebfun4j.util.ChebyshevPoints;
import com.marmanis.chebfun4j.util.Clenshaw;

import java.util.Arrays;

/**
 * A smooth-function representation on the canonical Chebyshev interval
 * {@code [-1, 1]}, storing the function as its <em>Chebyshev coefficients</em>
 * {@code c_0, ..., c_{n-1}} such that
 * {@code f(x) = sum_{k=0}^{n-1} c_k T_k(x)}. This mirrors MATLAB chebfun's
 * {@code chebtech2}, the single-piece unmapped building block that
 * {@link Chebfun} composes with a {@link Domain} to represent functions on
 * arbitrary {@code [a, b]}.
 *
 * <p>Coefficients are the canonical form here rather than the sampled values
 * (unlike upstream chebfun, which stores values and computes coefficients on
 * demand) because every operation this class implements — evaluation
 * (Clenshaw), integration ({@code sum}), differentiation ({@code diff}),
 * length reduction ({@code prolong}/{@code simplify}) — is either coefficient-
 * native or is at worst a single {@link ChebTransform#coeffs2vals} away.
 * Values are computed on demand by {@link #values()}.
 *
 * <p>Class is immutable and thread-safe; every returned {@code double[]} is a
 * defensive copy.
 */
public final class Chebtech implements Fun {
    private final double[] coeffs;

    private Chebtech(double[] coeffs) {
        this.coeffs = coeffs;
    }

    /** Length-{@code n} Chebtech from raw coefficients (defensive copy). */
    public static Chebtech fromCoeffs(double[] coeffs) {
        if (coeffs == null || coeffs.length == 0) {
            throw new IllegalArgumentException("coeffs must be non-empty");
        }
        return new Chebtech(coeffs.clone());
    }

    /**
     * Chebtech from values at the Chebyshev-2nd-kind grid of the same length.
     * Length must be 1 or {@code 2^k + 1} to hit the DCT-I fast path.
     */
    public static Chebtech fromValues(double[] values) {
        return new Chebtech(ChebTransform.vals2coeffs(values));
    }

    /** Constant Chebtech equal to {@code c} everywhere on {@code [-1, 1]}. */
    public static Chebtech constant(double c) {
        return new Chebtech(new double[]{c});
    }

    /** Number of coefficients (== grid length that {@link #values} returns). */
    public int length() {
        return coeffs.length;
    }

    /** Chebyshev coefficients {@code c_0, ..., c_{n-1}} (fresh copy). */
    public double[] coeffs() {
        return coeffs.clone();
    }

    /**
     * Values {@code f(x_j)} at the Chebyshev-2nd-kind grid of this Chebtech's
     * length (see {@link ChebyshevPoints#secondKind}). Recomputed from the
     * stored coefficients on every call — cache the result yourself if
     * you'll need it repeatedly.
     */
    public double[] values() {
        return ChebTransform.coeffs2vals(coeffs);
    }

    /**
     * Chebyshev-2nd-kind grid of length {@link #length()} on {@code [-1, 1]}
     * — i.e. the points at which {@link #values} is evaluated.
     */
    public double[] grid() {
        return ChebyshevPoints.secondKind(coeffs.length);
    }

    /** Evaluate {@code f(x)} for {@code x} in {@code [-1, 1]} via Clenshaw. */
    public double eval(double x) {
        return Clenshaw.eval(coeffs, x);
    }

    /**
     * Zero-pad the coefficient vector out to length {@code newLen}
     * ({@code newLen >= length()}). The mathematical function is unchanged;
     * this is used to bring two Chebtechs to the same length for pointwise
     * (value-based) arithmetic.
     */
    public Chebtech prolong(int newLen) {
        if (newLen < coeffs.length) {
            throw new IllegalArgumentException(
                "prolong requires newLen >= length(), got " + newLen + " < " + coeffs.length);
        }
        if (newLen == coeffs.length) return this;
        double[] extended = new double[newLen];
        System.arraycopy(coeffs, 0, extended, 0, coeffs.length);
        return new Chebtech(extended);
    }

    /**
     * Add two Chebtechs. Coefficient-wise after prolonging both to the max
     * of their lengths; the mathematical function is exactly the sum, no
     * value round-trip needed. The result is <em>not</em> simplified — the
     * caller (typically {@link Chebfun}) applies the tolerance-based simplify
     * once it knows the vscale.
     */
    public Chebtech plus(Chebtech other) {
        int n = Math.max(this.coeffs.length, other.coeffs.length);
        double[] out = new double[n];
        for (int k = 0; k < this.coeffs.length; k++) out[k] += this.coeffs[k];
        for (int k = 0; k < other.coeffs.length; k++) out[k] += other.coeffs[k];
        return new Chebtech(out);
    }

    /** Add a scalar (shifts {@code c_0}). */
    public Chebtech plus(double s) {
        double[] out = coeffs.clone();
        out[0] += s;
        return new Chebtech(out);
    }

    /** Subtract two Chebtechs; see {@link #plus(Chebtech)} for the shape rule. */
    public Chebtech minus(Chebtech other) {
        int n = Math.max(this.coeffs.length, other.coeffs.length);
        double[] out = new double[n];
        for (int k = 0; k < this.coeffs.length; k++) out[k] += this.coeffs[k];
        for (int k = 0; k < other.coeffs.length; k++) out[k] -= other.coeffs[k];
        return new Chebtech(out);
    }

    /** Unary negate. */
    public Chebtech negate() {
        double[] out = new double[coeffs.length];
        for (int k = 0; k < coeffs.length; k++) out[k] = -coeffs[k];
        return new Chebtech(out);
    }

    /** Scalar multiply. */
    public Chebtech times(double s) {
        double[] out = new double[coeffs.length];
        for (int k = 0; k < coeffs.length; k++) out[k] = s * coeffs[k];
        return new Chebtech(out);
    }

    /**
     * Multiply two Chebtechs. Product of degree-{@code m-1} and degree-
     * {@code n-1} polynomials has degree {@code m+n-2}, so the natural
     * output length is {@code m+n-1}; we round up to the next
     * {@link ChebyshevPoints#nextValidLength} to stay on the FFT fast path.
     * Computed in the value domain (pointwise multiply after
     * {@code coeffs2vals}, then transform back). Not simplified — caller
     * handles that with the combined vscale.
     */
    public Chebtech times(Chebtech other) {
        int naive = this.coeffs.length + other.coeffs.length - 1;
        int newLen = ChebyshevPoints.nextValidLength(naive);
        Chebtech a = this.prolong(newLen);
        Chebtech b = other.prolong(newLen);
        double[] va = a.values();
        double[] vb = b.values();
        double[] vc = new double[newLen];
        for (int i = 0; i < newLen; i++) vc[i] = va[i] * vb[i];
        return Chebtech.fromValues(vc);
    }

    /**
     * Definite integral over the reference interval {@code [-1, 1]}. Uses the
     * closed-form Chebyshev identity {@code integral(T_k) = 0} for odd
     * {@code k} and {@code 2 / (1 - k^2)} for even {@code k >= 2}
     * ({@code integral(T_0) = 2}). Exact to floating-point rounding for any
     * finite-degree polynomial.
     */
    public double sum() {
        int n = coeffs.length;
        double s = 2.0 * coeffs[0];
        for (int k = 2; k < n; k += 2) {
            s += 2.0 * coeffs[k] / (1.0 - (double) k * k);
        }
        return s;
    }

    /**
     * Derivative on the reference interval {@code [-1, 1]}. Output length is
     * {@code max(1, length() - 1)}. Uses the standard backward recurrence
     * on Chebyshev coefficients:
     * <pre>
     *   b_k = b_{k+2} + 2*(k+1)*c_{k+1},   k = n-2, n-3, ..., 0
     *   b_0 /= 2
     * </pre>
     * (with {@code b_{n-1} = b_n = 0}) — the same one MATLAB chebfun uses.
     */
    public Chebtech diff() {
        int n = coeffs.length;
        if (n <= 1) return new Chebtech(new double[]{0.0});
        double[] b = new double[n - 1];
        // Odd-indexed b's depend on odd-indexed c's; even on even. Two
        // independent backward cumulative sums with stride 2.
        // b_k = 2*(k+1)*c_{k+1} + (b_{k+2} if k+2 < n-1 else 0)
        for (int k = n - 2; k >= 0; k--) {
            double contribution = 2.0 * (k + 1) * coeffs[k + 1];
            double higher = (k + 2 <= n - 2) ? b[k + 2] : 0.0;
            b[k] = contribution + higher;
        }
        b[0] *= 0.5;
        return new Chebtech(b);
    }

    /**
     * Indefinite integral on the reference interval, with the constant of
     * integration chosen so that {@code F(-1) = 0}. Output length is
     * {@code length() + 1}. Uses the recurrence
     * {@code d_k = (c_{k-1} - c_{k+1}) / (2k)} for {@code k = 1..n} (treating
     * {@code c_n = c_{n+1} = 0}), then picks {@code d_0} to satisfy the
     * boundary condition.
     */
    public Chebtech cumsum() {
        int n = coeffs.length;
        double[] d = new double[n + 1];
        // T_1 has a distinct integration rule because integral(T_1) =
        //   T_2/4 + T_0/4, so d[1] gets c[0] whole rather than c[0]/2.
        double c2 = (n > 2) ? coeffs[2] : 0.0;
        d[1] = coeffs[0] - 0.5 * c2;
        for (int k = 2; k <= n; k++) {
            double cm = coeffs[k - 1];
            double cp = (k + 1 <= n - 1) ? coeffs[k + 1] : 0.0;
            d[k] = (cm - cp) / (2.0 * k);
        }
        // F(-1) = sum d_k * (-1)^k = 0 ⇒ d_0 = -sum_{k>=1} d_k * (-1)^k.
        double s = 0.0;
        for (int k = 1; k <= n; k++) {
            s += ((k & 1) == 0) ? d[k] : -d[k];
        }
        d[0] = -s;
        return new Chebtech(d);
    }

    /**
     * Real roots of this Chebtech in {@code [-1, 1]}, sorted ascending.
     * Delegates to {@link com.marmanis.chebfun4j.util.RootFinder} — the
     * colleague-matrix eigenvalue solve, with recursive subdivision above
     * degree {@code 100}.
     */
    @Override
    public double[] rootsOnRef() {
        return com.marmanis.chebfun4j.util.RootFinder.rootsOnRef(this);
    }

    /**
     * Drop trailing coefficients whose magnitude falls below
     * {@code tol * max(|c_k|)} — the standard happiness/plateau test. Never
     * drops below length 1. Returns {@code this} unchanged if no truncation
     * is possible.
     */
    public Chebtech simplify(double tol) {
        int n = coeffs.length;
        double maxAbs = 0.0;
        for (double c : coeffs) if (Math.abs(c) > maxAbs) maxAbs = Math.abs(c);
        if (maxAbs == 0.0) return new Chebtech(new double[]{0.0});
        double cutoff = tol * maxAbs;
        int newLen = n;
        while (newLen > 1 && Math.abs(coeffs[newLen - 1]) <= cutoff) newLen--;
        if (newLen == n) return this;
        return new Chebtech(Arrays.copyOf(coeffs, newLen));
    }
}
