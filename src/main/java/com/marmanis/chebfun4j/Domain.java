package com.marmanis.chebfun4j;

/**
 * Closed real interval {@code [a, b]} with {@code a < b}. The linear map
 * {@code y -> a + (b - a) * (y + 1) / 2} takes the canonical Chebyshev interval
 * {@code [-1, 1]} into this domain; {@link #fromRef(double)} does that
 * forward map, {@link #toRef(double)} its inverse. Chebfun stores its
 * coefficients in the reference interval and applies the map on evaluation,
 * so arithmetic between chebfuns is exact iff their domains agree.
 */
public record Domain(double a, double b) {
    public Domain {
        if (!(a < b)) {
            throw new IllegalArgumentException("Domain requires a < b, got [" + a + ", " + b + "]");
        }
    }

    /** {@code [-1, 1]}. */
    public static Domain unit() {
        return new Domain(-1.0, 1.0);
    }

    public double length() {
        return b - a;
    }

    /** Map from {@code [-1, 1]} to {@code [a, b]}. */
    public double fromRef(double y) {
        return a + (b - a) * (y + 1.0) * 0.5;
    }

    /** Map from {@code [a, b]} to {@code [-1, 1]}. */
    public double toRef(double x) {
        return (2.0 * x - (a + b)) / (b - a);
    }

    public boolean contains(double x) {
        return x >= a && x <= b;
    }

    public boolean equalsDomain(Domain other) {
        return a == other.a && b == other.b;
    }
}
