package com.marmanis.chebfun4j;

/**
 * A linear boundary condition applied at a single endpoint of a
 * {@link Chebop} problem. Sealed over the three classical kinds:
 * <ul>
 *   <li>{@link Dirichlet} — {@code u(endpoint) = value}</li>
 *   <li>{@link Neumann}   — {@code u'(endpoint) = value}</li>
 *   <li>{@link Robin}     — {@code alpha * u(endpoint) + beta * u'(endpoint) = value}</li>
 * </ul>
 *
 * <p>Dirichlet and Neumann are just the {@code (1, 0)} and {@code (0, 1)}
 * cases of Robin respectively, but keeping them as distinct variants
 * makes call sites read the way the math reads and lets Chebop's row
 * assembly skip a multiply-by-zero.
 *
 * <p>The BC's {@code endpoint} is implicit — {@code Chebop.solve} takes a
 * pair {@code (bcA, bcB)} where {@code bcA} is applied at the left
 * endpoint {@code a} of the domain and {@code bcB} at the right endpoint
 * {@code b}. Which side a given BC is on is thus positional, not encoded
 * in the {@code BoundaryCondition} itself.
 */
public sealed interface BoundaryCondition {
    /** Right-hand side value the BC enforces. */
    double value();

    /**
     * {@code u(endpoint) = value}.
     */
    record Dirichlet(double value) implements BoundaryCondition {}

    /**
     * {@code u'(endpoint) = value}. Zero-value Neumann ({@code u'(endpoint) = 0})
     * is the natural "no flux" / insulated BC.
     */
    record Neumann(double value) implements BoundaryCondition {}

    /**
     * {@code alpha * u(endpoint) + beta * u'(endpoint) = value}. Reduces to
     * Dirichlet when {@code beta == 0} and to Neumann when {@code alpha == 0}
     * — but the caller should just use those variants when they apply. A
     * Robin BC with both coefficients zero is rejected.
     */
    record Robin(double alpha, double beta, double value) implements BoundaryCondition {
        public Robin {
            if (alpha == 0.0 && beta == 0.0) {
                throw new IllegalArgumentException(
                    "Robin BC requires at least one of alpha, beta to be non-zero");
            }
        }
    }
}
