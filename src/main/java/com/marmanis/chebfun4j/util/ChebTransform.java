package com.marmanis.chebfun4j.util;

import com.marmanis.jax4j.api.Fft;

/**
 * Values ↔ Chebyshev coefficients transforms for the second-kind grid,
 * layered on top of {@link Fft#dctIRaw(double[])}.
 *
 * <p>Given {@code N} values {@code v_j = f(x_j)} at the second-kind points
 * {@code x_j = cos(pi j / (N-1))} for {@code j = 0..N-1} (see
 * {@link ChebyshevPoints#secondKind}), the Chebyshev-series coefficients
 * {@code c_k} such that {@code f(x) ~= sum_{k=0}^{N-1} c_k T_k(x)} are:
 * <pre>
 *   c_k = (1 / (N-1)) * DCT-I(v)_k,   1 &lt;= k &lt;= N-2
 *   c_0 = (1 / (2(N-1))) * DCT-I(v)_0
 *   c_{N-1} = (1 / (2(N-1))) * DCT-I(v)_{N-1}
 * </pre>
 * (using the unnormalized DCT-I convention of {@link Fft#dctIRaw}). The
 * inverse uses the exact same DCT-I after doubling the endpoint
 * coefficients and dividing by 2 — the transform is its own inverse up to
 * a straightforward boundary rescaling. Both directions require
 * {@code N == 1} or {@code N - 1} to be a power of two, matching the grid
 * sizes chebfun's adaptive constructor tries.
 *
 * <p>This is chebfun4j's hottest path: every Chebtech construction and
 * every {@code Chebtech.times} calls it. The raw-{@code double[]} DCT-I
 * lets us skip the {@code NDArray}/{@code Shape} allocation on every call.
 */
public final class ChebTransform {
    private ChebTransform() {}

    /**
     * Convert function values at Chebyshev-2nd-kind points to Chebyshev
     * coefficients. Input length must be 1 or {@code 2^k + 1}.
     */
    public static double[] vals2coeffs(double[] values) {
        int n = values.length;
        if (n == 0) throw new IllegalArgumentException("empty values");
        if (n == 1) return new double[]{values[0]};
        double[] y = Fft.dctIRaw(values);
        double invM = 1.0 / (n - 1);
        double[] c = new double[n];
        c[0] = 0.5 * invM * y[0];
        for (int k = 1; k < n - 1; k++) c[k] = invM * y[k];
        c[n - 1] = 0.5 * invM * y[n - 1];
        return c;
    }

    /**
     * Convert Chebyshev coefficients to values at the Chebyshev-2nd-kind
     * points. Input length must be 1 or {@code 2^k + 1}.
     */
    public static double[] coeffs2vals(double[] coeffs) {
        int n = coeffs.length;
        if (n == 0) throw new IllegalArgumentException("empty coeffs");
        if (n == 1) return new double[]{coeffs[0]};
        double[] scaled = new double[n];
        scaled[0] = 2.0 * coeffs[0];
        System.arraycopy(coeffs, 1, scaled, 1, n - 2);
        scaled[n - 1] = 2.0 * coeffs[n - 1];
        double[] y = Fft.dctIRaw(scaled);
        double[] v = new double[n];
        for (int j = 0; j < n; j++) v[j] = 0.5 * y[j];
        return v;
    }
}
