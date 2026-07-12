package com.marmanis.chebfun4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;

/**
 * A smooth <em>periodic</em> function on the reference interval
 * {@code [-1, 1]} (period {@code 2}), stored as its truncated Fourier
 * series
 * {@code g(y) = a_0 + sum_{k=1..N/2-1} (a_k cos(pi k y) + b_k sin(pi k y))
 *              + a_{N/2} cos(pi (N/2) y)}
 * where {@code N} is the sample length (a power of two, so that
 * {@link Fft#rfft} can produce the coefficients). This is the periodic
 * sister of {@link Chebtech}, mirroring MATLAB chebfun's {@code trigtech}.
 *
 * <p>Implements {@link Fun}, so a {@link Trigfun} (or, eventually, a
 * piecewise {@link Chebfun} with some Trigtech pieces) can use the same
 * container machinery as chebtech pieces do.
 *
 * <p>The reference interval {@code [-1, 1]} is used, rather than
 * {@code [0, 2 pi]}, so that {@link Fun#eval(double)} composes with
 * {@link Domain}'s linear map exactly as {@link Chebtech} does — the two
 * bases become interchangeable at the container level.
 */
public final class Trigtech implements Fun {
    private final double[] a; // cosine coefficients, indices 0..N/2
    private final double[] b; // sine coefficients, indices 1..N/2-1 (0 and N/2 slots unused, kept for symmetric indexing)

    private Trigtech(double[] a, double[] b) {
        this.a = a;
        this.b = b;
    }

    /**
     * Length of the underlying sample vector — the {@code N} of the Fourier
     * expansion. Always a power of two (or {@code 1} for the constant
     * degenerate case).
     */
    @Override
    public int length() {
        // a has N/2 + 1 entries; total sample length is N.
        int m = a.length; // = N/2 + 1
        return (m == 1) ? 1 : 2 * (m - 1);
    }

    /** Cosine coefficients {@code a_0, a_1, ..., a_{N/2}}. */
    public double[] cosCoeffs() { return a.clone(); }

    /** Sine coefficients (index 0 and {@code N/2} are zero by convention). */
    public double[] sinCoeffs() { return b.clone(); }

    /**
     * Constant trigtech equal to {@code c} everywhere on {@code [-1, 1]}.
     */
    public static Trigtech constant(double c) {
        return new Trigtech(new double[]{c}, new double[]{0.0});
    }

    /**
     * Trigtech from real values at the equispaced grid
     * {@code y_j = -1 + 2 j / N} for {@code j = 0..N-1}. {@code N} must be a
     * power of two. The grid does <em>not</em> include the right endpoint
     * {@code y = 1} — that's the "periodic" grid convention (the value at
     * {@code y = 1} equals the value at {@code y = -1} by periodicity, so
     * it would be redundant).
     */
    public static Trigtech fromValues(double[] values) {
        int n = values.length;
        if (n < 1) throw new IllegalArgumentException("empty values");
        if (n == 1) return constant(values[0]);
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException(
                "trigtech length must be a power of two, got " + n);
        }
        NDArray input = new ConcreteNDArray(values.clone(), new Shape(n));
        NDArray[] spec = Fft.rfft(input);
        double[] specRe = spec[0].toDoubleArray();
        double[] specIm = spec[1].toDoubleArray();
        int half = specRe.length; // = N/2 + 1
        double[] aCoeffs = new double[half];
        double[] bCoeffs = new double[half];
        // Our sample grid starts at y = -1, which introduces a per-bin
        // phase factor of (-1)^k relative to the standard DFT convention
        // (which treats sample 0 as the origin). Absorb the phase into the
        // real/imag pickup so the resulting Fourier coefficients are what
        // g(y) = a_0 + sum a_k cos(pi k y) + b_k sin(pi k y) + Nyquist
        // literally computes back.
        aCoeffs[0] = specRe[0] / n;
        for (int k = 1; k < half - 1; k++) {
            double phase = ((k & 1) == 0) ? 1.0 : -1.0;
            aCoeffs[k] =  2.0 * phase * specRe[k] / n;
            bCoeffs[k] = -2.0 * phase * specIm[k] / n;
        }
        int nyq = half - 1;
        double nyqPhase = ((nyq & 1) == 0) ? 1.0 : -1.0;
        aCoeffs[nyq] = nyqPhase * specRe[nyq] / n; // Nyquist: cos(pi (N/2) y) only
        return new Trigtech(aCoeffs, bCoeffs);
    }

    @Override
    public double eval(double y) {
        int m = a.length;
        if (m == 1) return a[0];
        double sum = a[0];
        for (int k = 1; k < m - 1; k++) {
            double theta = Math.PI * k * y;
            sum += a[k] * Math.cos(theta) + b[k] * Math.sin(theta);
        }
        // Nyquist term is cos only.
        sum += a[m - 1] * Math.cos(Math.PI * (m - 1) * y);
        return sum;
    }

    @Override
    public Trigtech negate() {
        double[] na = new double[a.length];
        double[] nb = new double[b.length];
        for (int i = 0; i < a.length; i++) na[i] = -a[i];
        for (int i = 0; i < b.length; i++) nb[i] = -b[i];
        return new Trigtech(na, nb);
    }

    @Override
    public Trigtech times(double s) {
        double[] na = new double[a.length];
        double[] nb = new double[b.length];
        for (int i = 0; i < a.length; i++) na[i] = s * a[i];
        for (int i = 0; i < b.length; i++) nb[i] = s * b[i];
        return new Trigtech(na, nb);
    }

    /**
     * Derivative with respect to the reference variable {@code y}.
     * {@code (d/dy) cos(pi k y) = -pi k sin(pi k y)}, so
     * new {@code a_k = pi k b_k}, new {@code b_k = -pi k a_k}; the
     * constant term vanishes and the Nyquist cosine term becomes a sine
     * (i.e. cannot be represented in the same length — we drop it, which
     * is the standard convention for even-{@code N} Fourier
     * differentiation).
     */
    @Override
    public Trigtech diff() {
        int m = a.length;
        double[] na = new double[m];
        double[] nb = new double[m];
        for (int k = 1; k < m - 1; k++) {
            na[k] =  Math.PI * k * b[k];
            nb[k] = -Math.PI * k * a[k];
        }
        // Nyquist -> sine that we cannot store at the same N. Convention:
        // set both na[N/2] and nb[N/2] to zero (equivalent to zeroing the
        // Nyquist mode before differentiating).
        return new Trigtech(na, nb);
    }

    /**
     * Indefinite integral on the reference. The constant of integration
     * is chosen so that the mean of the antiderivative over one period is
     * zero — the natural convention for periodic antiderivatives. This is
     * NOT the same as {@code F(-1) = 0}, so composing this with
     * {@link Chebfun}'s piecewise cumsum would need care; for MVP
     * {@link Trigfun#cumsum()} handles that shift itself.
     */
    @Override
    public Trigtech cumsum() {
        int m = a.length;
        // If a_0 != 0, the integral has a linear-in-y term and is NOT
        // periodic. We refuse rather than silently producing a
        // representation that is not what it claims to be.
        if (Math.abs(a[0]) > 1e-14 * vscaleGuess()) {
            throw new IllegalStateException(
                "cumsum on a non-mean-zero periodic function is not periodic; " +
                "constant term a_0 = " + a[0] + " must vanish");
        }
        double[] na = new double[m];
        double[] nb = new double[m];
        for (int k = 1; k < m - 1; k++) {
            na[k] = -b[k] / (Math.PI * k);
            nb[k] =  a[k] / (Math.PI * k);
        }
        // Nyquist term contributes similarly; we handle it symmetrically.
        int nyq = m - 1;
        if (nyq > 0) nb[nyq] = a[nyq] / (Math.PI * nyq);
        na[0] = 0.0; // mean-zero antiderivative
        return new Trigtech(na, nb);
    }

    private double vscaleGuess() {
        double v = 0.0;
        for (double x : a) v = Math.max(v, Math.abs(x));
        for (double x : b) v = Math.max(v, Math.abs(x));
        return Math.max(v, 1.0);
    }

    /**
     * Definite integral over one period {@code [-1, 1]}: only the constant
     * term survives, and integrates to {@code 2 a_0}.
     */
    @Override
    public double sum() {
        return 2.0 * a[0];
    }

    /**
     * Real roots of the periodic function in {@code [-1, 1)}. Uses sign-
     * change bracketing on a fine equispaced grid plus a few bisection
     * refinements. Adequate for simple crossings; misses even-multiplicity
     * roots by design (rare for periodic-function use cases; upgrade to
     * a companion-matrix approach if that becomes a real need).
     */
    @Override
    public double[] rootsOnRef() {
        int samples = Math.max(4 * length(), 65);
        double[] xs = new double[samples];
        double[] fs = new double[samples];
        double dx = 2.0 / (samples - 1);
        for (int i = 0; i < samples; i++) {
            xs[i] = -1.0 + i * dx;
            fs[i] = eval(xs[i]);
        }
        java.util.List<Double> roots = new java.util.ArrayList<>();
        for (int i = 0; i < samples - 1; i++) {
            if (fs[i] == 0.0) { roots.add(xs[i]); continue; }
            if (fs[i] * fs[i + 1] < 0.0) {
                roots.add(bisect(xs[i], xs[i + 1], fs[i], fs[i + 1]));
            }
        }
        double[] out = new double[roots.size()];
        for (int i = 0; i < out.length; i++) out[i] = roots.get(i);
        return out;
    }

    private double bisect(double lo, double hi, double fLo, double fHi) {
        for (int i = 0; i < 60; i++) {
            double mid = 0.5 * (lo + hi);
            double fm = eval(mid);
            if (fm == 0.0 || (hi - lo) < 1e-15) return mid;
            if (fLo * fm < 0) { hi = mid; fHi = fm; }
            else { lo = mid; fLo = fm; }
        }
        return 0.5 * (lo + hi);
    }

    /**
     * Drop trailing Fourier coefficients whose magnitude falls below
     * {@code tol * max(|a_k|, |b_k|)}. Reduces the sample length by
     * halving until a coefficient above threshold is retained — Fourier
     * length must stay a power of two.
     */
    @Override
    public Trigtech simplify(double tol) {
        int m = a.length;
        int n = length();
        if (n <= 1) return this;
        double maxAbs = 0.0;
        for (double v : a) maxAbs = Math.max(maxAbs, Math.abs(v));
        for (double v : b) maxAbs = Math.max(maxAbs, Math.abs(v));
        if (maxAbs == 0.0) return constant(0.0);
        double cutoff = tol * maxAbs;
        int highestCos = 0;
        int highestSin = 0;
        for (int k = 0; k < m; k++) {
            if (Math.abs(a[k]) > cutoff) highestCos = k;
            if (Math.abs(b[k]) > cutoff) highestSin = k;
        }
        // Nyquist (index N/2) can hold a cos-only mode, so it's OK for
        // highestCos to equal N/2; but a sine mode at index k requires
        // k < N/2 (strictly below Nyquist), so we need N/2 >= highestSin + 1
        // whenever there IS a sine coefficient — otherwise it gets dropped.
        int needHalfIndex = Math.max(highestCos, highestSin > 0 ? highestSin + 1 : 0);
        // Smallest power-of-two N with N/2 >= needHalfIndex.
        int newN = 2;
        while (newN / 2 < needHalfIndex) newN <<= 1;
        if (newN >= n) return this;
        int newHalf = newN / 2 + 1;
        double[] na = new double[newHalf];
        double[] nb = new double[newHalf];
        int copyLen = Math.min(newHalf, m);
        System.arraycopy(a, 0, na, 0, copyLen);
        System.arraycopy(b, 0, nb, 0, copyLen);
        return new Trigtech(na, nb);
    }

    /** Add two Trigtechs; both padded to the max length. */
    public Trigtech plus(Trigtech other) {
        int m = Math.max(this.a.length, other.a.length);
        double[] na = new double[m];
        double[] nb = new double[m];
        for (int k = 0; k < this.a.length; k++)  { na[k] += this.a[k];  nb[k] += this.b[k]; }
        for (int k = 0; k < other.a.length; k++) { na[k] += other.a[k]; nb[k] += other.b[k]; }
        return new Trigtech(na, nb);
    }

    /** Subtract two Trigtechs. */
    public Trigtech minus(Trigtech other) {
        int m = Math.max(this.a.length, other.a.length);
        double[] na = new double[m];
        double[] nb = new double[m];
        for (int k = 0; k < this.a.length; k++)  { na[k] += this.a[k];  nb[k] += this.b[k]; }
        for (int k = 0; k < other.a.length; k++) { na[k] -= other.a[k]; nb[k] -= other.b[k]; }
        return new Trigtech(na, nb);
    }

    /**
     * Multiply two Trigtechs by pointwise product in the value domain: sample
     * both at the same larger grid, multiply, transform back. Length is chosen
     * so the exact product's aliasing is avoided (product of degree-{@code n},
     * degree-{@code m} trigonometric polynomials has degree {@code n + m}).
     */
    public Trigtech times(Trigtech other) {
        int n1 = this.length();
        int n2 = other.length();
        int need = n1 + n2;
        int newN = 1;
        while (newN < need) newN <<= 1;
        newN = Math.max(newN, 2);
        double[] va = valuesOn(newN);
        double[] vb = other.valuesOn(newN);
        double[] vc = new double[newN];
        for (int i = 0; i < newN; i++) vc[i] = va[i] * vb[i];
        return fromValues(vc);
    }

    /**
     * Sample this Trigtech on an {@code n}-point equispaced grid (n a power
     * of two). Uses evaluation; a faster path via zero-padded ifft is
     * possible but the eval path is O(n * m) which is fine for MVP.
     */
    public double[] valuesOn(int n) {
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("valuesOn length must be a power of two, got " + n);
        }
        double[] out = new double[n];
        double dx = 2.0 / n;
        for (int j = 0; j < n; j++) out[j] = eval(-1.0 + j * dx);
        return out;
    }
}
