package com.marmanis.chebfun4j;

/**
 * The Cartesian product of two closed real intervals, {@code [xa, xb] × [ya, yb]}
 * with {@code xa < xb} and {@code ya < yb}. Used by {@link Chebfun2} as the
 * 2-D counterpart of {@link Domain}; the linear map from the canonical
 * reference {@code [-1, 1] × [-1, 1]} to the physical rectangle is
 * separable, so {@link Domain} handles each axis on its own.
 */
public record Rectangle(double xa, double xb, double ya, double yb) {
    public Rectangle {
        if (!(xa < xb) || !(ya < yb)) {
            throw new IllegalArgumentException(
                "Rectangle requires xa < xb and ya < yb, got [" + xa + ", " + xb +
                "] x [" + ya + ", " + yb + "]");
        }
    }

    /** {@code [-1, 1] × [-1, 1]}. */
    public static Rectangle unit() {
        return new Rectangle(-1.0, 1.0, -1.0, 1.0);
    }

    public Domain xDomain() { return new Domain(xa, xb); }
    public Domain yDomain() { return new Domain(ya, yb); }

    public double width()  { return xb - xa; }
    public double height() { return yb - ya; }

    /** {@code true} iff {@code (x, y)} lies inside the rectangle. */
    public boolean contains(double x, double y) {
        return x >= xa && x <= xb && y >= ya && y <= yb;
    }

    public boolean equalsRectangle(Rectangle other) {
        return xa == other.xa && xb == other.xb && ya == other.ya && yb == other.yb;
    }
}
