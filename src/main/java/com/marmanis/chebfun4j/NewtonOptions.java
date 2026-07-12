package com.marmanis.chebfun4j;

/**
 * Configuration for Newton iteration in {@link NonlinearChebop}. Fields
 * cover the standard knobs:
 * <ul>
 *   <li>{@code maxIter} — maximum Newton iterations per grid size before
 *       we escalate the discretization (or give up).</li>
 *   <li>{@code tol} — convergence tolerance on the max-abs Newton update
 *       relative to the current solution's vscale.</li>
 *   <li>{@code initialDamping} — starting damping factor for the step;
 *       {@code 1.0} is full Newton, smaller values are conservative. On
 *       stalled iterations we halve this via an Armijo-style backtrack.</li>
 *   <li>{@code initialGuess} — starting guess for {@code u}, or
 *       {@code null} to seed with zero.</li>
 * </ul>
 *
 * <p>{@link #defaults()} returns a sensible starting point for smooth
 * nonlinear BVPs.
 */
public record NewtonOptions(int maxIter,
                            double tol,
                            double initialDamping,
                            Chebfun initialGuess) {

    public NewtonOptions {
        if (maxIter < 1) throw new IllegalArgumentException("maxIter must be >= 1");
        if (!(tol > 0)) throw new IllegalArgumentException("tol must be positive");
        if (!(initialDamping > 0 && initialDamping <= 1))
            throw new IllegalArgumentException("initialDamping must be in (0, 1]");
    }

    /**
     * {@code maxIter = 30}, {@code tol = 1e-10}, {@code initialDamping = 1.0},
     * {@code initialGuess = null} (zero seed).
     */
    public static NewtonOptions defaults() {
        return new NewtonOptions(30, 1e-10, 1.0, null);
    }

    /** With a specific initial guess (other fields default). */
    public static NewtonOptions withInitialGuess(Chebfun guess) {
        return new NewtonOptions(30, 1e-10, 1.0, guess);
    }
}
