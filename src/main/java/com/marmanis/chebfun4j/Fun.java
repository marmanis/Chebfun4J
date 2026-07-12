package com.marmanis.chebfun4j;

/**
 * A smooth-function representation on the canonical reference interval
 * {@code [-1, 1]}. This is the shared abstraction over
 * {@link Chebtech} (polynomial series in the Chebyshev basis) and future
 * periodic representations ({@code Trigtech}, Fourier basis) — what MATLAB
 * chebfun calls the {@code onefun} layer.
 *
 * <p>{@link Chebfun} composes {@code Fun} pieces with a domain map and a
 * list of breakpoints to represent piecewise-smooth functions on
 * {@code [a, b]}. Same-type binary arithmetic (adding two Chebtechs,
 * multiplying two Chebtechs) stays on the concrete class where it can
 * exploit basis-specific formulas; the piecewise container is careful
 * only to combine pieces of the same runtime type and will refuse
 * (with a clear error) to mix bases.
 */
public interface Fun {
    /** Number of coefficients (or samples) representing this function. */
    int length();

    /** Evaluate {@code f(x)} for {@code x} in {@code [-1, 1]}. */
    double eval(double x);

    /** {@code -f}. Same concrete type as {@code this}. */
    Fun negate();

    /** {@code s * f} for a scalar {@code s}. Same concrete type as {@code this}. */
    Fun times(double s);

    /**
     * Derivative {@code f'} on the reference interval. Same concrete type as
     * {@code this}. The chain-rule factor for a non-unit domain is applied
     * by {@link Chebfun}.
     */
    Fun diff();

    /**
     * Indefinite integral {@code F} such that {@code F(-1) = 0}. Same
     * concrete type as {@code this}.
     */
    Fun cumsum();

    /** Definite integral over {@code [-1, 1]}. */
    double sum();

    /**
     * Real roots of {@code f} in {@code [-1, 1]}, sorted ascending. Empty
     * for a constant {@code Fun} regardless of whether it is identically
     * zero.
     */
    double[] rootsOnRef();

    /**
     * Drop insignificant trailing coefficients (or high-frequency modes)
     * per the concrete type's plateau/happiness test. Returns {@code this}
     * unchanged if no simplification is possible.
     */
    Fun simplify(double tol);
}
