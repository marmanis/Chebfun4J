package com.marmanis.chebfun4j;

/**
 * A single scalar linear boundary condition on one component of a
 * multi-component system solved by {@link LinearBlockChebop} (or a future
 * NonlinearSystem). Each variant identifies the component (0-indexed)
 * the BC applies to and the right-hand-side value it enforces:
 * <ul>
 *   <li>{@link Dirichlet} — {@code u_c(endpoint) = value}</li>
 *   <li>{@link Neumann}   — {@code u_c'(endpoint) = value}</li>
 *   <li>{@link Robin}     — {@code alpha * u_c(endpoint) + beta * u_c'(endpoint) = value}</li>
 * </ul>
 *
 * <p>A k-component 2nd-order system needs {@code 2k} boundary conditions
 * total. The distribution across the two endpoints is fixed by the caller:
 * {@code LinearBlockChebop.solve} takes two arrays, one per endpoint,
 * and the sum of their lengths must equal the required BC count.
 *
 * <p>The scalar {@link BoundaryCondition} used by {@link Chebop} is the
 * degenerate {@code k = 1} case (component index is always {@code 0}).
 */
public sealed interface SystemBC {
    /** 0-indexed component of the system this BC applies to. */
    int component();
    /** Right-hand-side value the BC enforces. */
    double value();

    /** {@code u_c(endpoint) = value}. */
    record Dirichlet(int component, double value) implements SystemBC {}

    /** {@code u_c'(endpoint) = value}. */
    record Neumann(int component, double value) implements SystemBC {}

    /**
     * {@code alpha * u_c(endpoint) + beta * u_c'(endpoint) = value}. At
     * least one of {@code alpha}, {@code beta} must be non-zero (else
     * the BC is either identically satisfied or contradictory).
     */
    record Robin(int component, double alpha, double beta, double value) implements SystemBC {
        public Robin {
            if (alpha == 0.0 && beta == 0.0) {
                throw new IllegalArgumentException(
                    "Robin BC requires at least one of alpha, beta to be non-zero");
            }
        }
    }
}
