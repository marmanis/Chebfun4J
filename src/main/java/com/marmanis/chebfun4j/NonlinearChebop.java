package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.util.ArrayMath;
import com.marmanis.chebfun4j.util.DifferentiationMatrix;
import com.marmanis.jax4j.api.JAX;
import com.marmanis.jax4j.api.Linalg;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import com.marmanis.jax4j.api.Grad;
import com.marmanis.jax4j.ir.Jaxpr;
import com.marmanis.jax4j.ir.Var;
import com.marmanis.jax4j.pytree.PyTree;
import com.marmanis.jax4j.tracing.TracedNDArray;
import com.marmanis.jax4j.tracing.Tracer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

/**
 * A nonlinear scalar ODE boundary-value problem
 * <pre>
 *   F(x, u(x), u'(x), u''(x)) = 0,  on [a, b], with linear BCs.
 * </pre>
 * User specifies the residual {@code F} pointwise as a callable
 * {@code (x, u, u', u'') -> residual}. Boundary conditions are the same
 * linear {@link BoundaryCondition} type as {@link Chebop} — nonlinear
 * BCs are not supported.
 *
 * <p><strong>Discretization + Newton iteration.</strong> On a Chebyshev
 * second-kind grid of size {@code n+1}, {@code u} is represented by its
 * values on the grid, and {@code u'}, {@code u''} are approximated via
 * the spectral differentiation matrix {@code D}. The Newton step at each
 * iterate solves the linearized system {@code J Δu = -r}, where
 * <pre>
 *   J[i][j] = ∂F/∂u  * δ_{ij}
 *           + ∂F/∂u' * D[i][j]
 *           + ∂F/∂u'' * D²[i][j],
 * </pre>
 * with the three partials evaluated at the current iterate. Users can
 * override {@link Residual} to supply analytic partials — with a jax4j
 * autodiff-based residual, the partials are cheap and exact; the default
 * uses centered finite differences on {@code F} itself (good to about
 * {@code sqrt(eps) ≈ 1e-8}, plenty for Newton's quadratic convergence).
 *
 * <p><strong>Damped Newton (Armijo).</strong> If the full Newton step
 * increases the residual norm, we backtrack: try step size
 * {@code s/2}, {@code s/4}, ... until the norm decreases (or five
 * backtracks, then accept and continue with a warning).
 *
 * <p><strong>Adaptive grid.</strong> Try grid sizes {@code 16, 32, ...,
 * 512} in sequence. Each grid seeds Newton with the previous grid's
 * solution (interpolated to the new grid), so the outer loop performs a
 * continuation in {@code n}.
 */
public final class NonlinearChebop {

    /**
     * Pointwise residual for a nonlinear ODE:
     * {@code F(x, u, u', u'') = 0}. The residual is evaluated
     * <em>point by point</em> — {@code F_i} at grid index {@code i} may
     * depend only on {@code x_i}, {@code u_i}, {@code u'_i}, {@code u''_i}
     * (not on any other grid index). This locality is what lets the
     * Jacobian be assembled from the three per-point partial derivatives.
     *
     * <p>The core scalar API ({@link #at}, {@link #dU}, {@link #dUp},
     * {@link #dUpp}) is the simplest way to specify a residual. For
     * performance the Newton loop actually calls the {@link Batched}
     * view; the default {@link #batched(int)} implementation scalar-loops
     * over every grid point, but the autodiff wrapper overrides it to
     * evaluate all {@code n+1} residuals and partial derivatives at once
     * (three vector {@code JAX.grad} traces per Newton iteration instead
     * of {@code 3 * (n+1)} scalar traces).
     */
    public interface Residual {
        /** {@code F(x, u, u', u'')}. Must be smooth in the last three arguments. */
        double at(double x, double u, double up, double upp);

        /**
         * Partial derivative {@code ∂F/∂u}. Default: centered finite
         * difference of {@link #at}. Override for analytic accuracy, or
         * use {@link NonlinearChebop#autodiffResidual} to get exact
         * partials via jax4j's reverse-mode autodiff.
         */
        default double dU(double x, double u, double up, double upp) {
            double h = fdStep(u);
            return (at(x, u + h, up, upp) - at(x, u - h, up, upp)) / (2 * h);
        }

        /** Partial {@code ∂F/∂u'}. */
        default double dUp(double x, double u, double up, double upp) {
            double h = fdStep(up);
            return (at(x, u, up + h, upp) - at(x, u, up - h, upp)) / (2 * h);
        }

        /** Partial {@code ∂F/∂u''}. */
        default double dUpp(double x, double u, double up, double upp) {
            double h = fdStep(upp);
            return (at(x, u, up, upp + h) - at(x, u, up, upp - h)) / (2 * h);
        }

        /**
         * A view of this residual that evaluates all grid points at once.
         * The Newton loop uses this batched view; the default falls back
         * to a scalar loop, but autodiff residuals override with a
         * vectorized implementation.
         */
        default Batched batched(int size) {
            return new ScalarLoopBatched(this, size);
        }

        private static double fdStep(double x) {
            return Math.cbrt(Math.ulp(1.0)) * Math.max(1.0, Math.abs(x));
        }
    }

    /**
     * Batched view of a {@link Residual}: evaluate the residual and its
     * three partial derivatives at all grid points at once. All inputs
     * and outputs are FLOAT64 vectors of the grid length. Autodiff
     * implementations override this to run {@code 3} traces per Newton
     * iteration (one per partial) instead of {@code 3 * (n+1)} traces.
     */
    public interface Batched {
        /** {@code F(x_i, u_i, u'_i, u''_i)} for all {@code i}, returned as a fresh {@code double[]}. */
        double[] residuals(double[] x, double[] u, double[] up, double[] upp);

        /** {@code ∂F/∂u} evaluated at every grid point. */
        double[] dU(double[] x, double[] u, double[] up, double[] upp);

        /** {@code ∂F/∂u'} evaluated at every grid point. */
        double[] dUp(double[] x, double[] u, double[] up, double[] upp);

        /** {@code ∂F/∂u''} evaluated at every grid point. */
        double[] dUpp(double[] x, double[] u, double[] up, double[] upp);

        /**
         * All three partial derivative vectors at once, in
         * {@code {dU, dUp, dUpp}} order. Default: fan out to the three
         * scalar methods. Batched implementations that compute all
         * partials together (e.g. the autodiff variant that runs one
         * pytree grad and reads three leaves) override this to skip the
         * redundant work.
         */
        default double[][] allPartials(double[] x, double[] u, double[] up, double[] upp) {
            return new double[][]{
                dU  (x, u, up, upp),
                dUp (x, u, up, upp),
                dUpp(x, u, up, upp)
            };
        }
    }

    /** Scalar-loop {@link Batched} — delegates to the scalar {@link Residual} methods. */
    private static final class ScalarLoopBatched implements Batched {
        private final Residual r;
        ScalarLoopBatched(Residual r, int ignoredSize) { this.r = r; }
        @Override
        public double[] residuals(double[] x, double[] u, double[] up, double[] upp) {
            double[] out = new double[u.length];
            for (int i = 0; i < u.length; i++) out[i] = r.at(x[i], u[i], up[i], upp[i]);
            return out;
        }
        @Override
        public double[] dU(double[] x, double[] u, double[] up, double[] upp) {
            double[] out = new double[u.length];
            for (int i = 0; i < u.length; i++) out[i] = r.dU(x[i], u[i], up[i], upp[i]);
            return out;
        }
        @Override
        public double[] dUp(double[] x, double[] u, double[] up, double[] upp) {
            double[] out = new double[u.length];
            for (int i = 0; i < u.length; i++) out[i] = r.dUp(x[i], u[i], up[i], upp[i]);
            return out;
        }
        @Override
        public double[] dUpp(double[] x, double[] u, double[] up, double[] upp) {
            double[] out = new double[u.length];
            for (int i = 0; i < u.length; i++) out[i] = r.dUpp(x[i], u[i], up[i], upp[i]);
            return out;
        }
    }

    /**
     * A residual expressed in terms of jax4j {@link NDArray} operations —
     * the "autodiff DSL" form. Every argument is a FLOAT64 NDArray of the
     * same shape (scalar for a pointwise call, or a length-{@code n+1}
     * vector when the batched view evaluates all grid points at once).
     * Because every operation on the way in is a jax4j primitive
     * (elementwise arithmetic, exp, sin, etc.), the same lambda works in
     * either mode without change.
     *
     * <p>Example: {@code (x, u, up, upp) -> upp.sub(u.mul(u))} for
     * {@code F = u'' - u^2}. For a constant coefficient inside the
     * expression, wrap it with {@link NonlinearChebop#scalar(double)}:
     * {@code (x, u, up, upp) -> upp.add(u.exp().mul(scalar(lambda)))}
     * for Bratu {@code u'' + λ e^u}.
     */
    @FunctionalInterface
    public interface AutodiffFn {
        NDArray apply(NDArray x, NDArray u, NDArray up, NDArray upp);
    }

    /**
     * Wrap an {@link AutodiffFn} as a {@link Residual} whose partial
     * derivatives are computed by jax4j reverse-mode autodiff.
     *
     * <p>The returned {@link Residual} exposes the standard scalar API
     * (for compatibility with hand-written {@code Residual}s) but its
     * {@link Residual#batched(int)} view is vectorized: each of the three
     * partial derivatives is computed by a single {@code JAX.grad} call
     * on {@code sum(F)} interpreted as a function of that variable alone.
     * For a length-{@code n+1} grid, this replaces
     * {@code 3 * (n+1)} scalar traces with {@code 3} vector traces per
     * Newton iteration — the whole point of batching.
     *
     * <p>Correctness of the "sum trick" relies on the pointwise
     * assumption: {@code F_i} depends only on {@code (x_i, u_i, u'_i,
     * u''_i)}. Then {@code ∂(sum_j F_j)/∂u_i = ∂F_i/∂u_i} because the
     * cross terms vanish, so a scalar-output {@code JAX.grad} directly
     * yields the diagonal of the Jacobian.
     */
    public static Residual autodiffResidual(AutodiffFn fn) {
        return new Residual() {
            @Override
            public double at(double x, double u, double up, double upp) {
                NDArray r = fn.apply(scalarF64(x), scalarF64(u), scalarF64(up), scalarF64(upp));
                return firstDouble(r);
            }
            @Override
            public double dU(double x, double u, double up, double upp) {
                final NDArray xC = scalarF64(x), upC = scalarF64(up), uppC = scalarF64(upp);
                Function<NDArray, NDArray> partial = uVar -> fn.apply(xC, uVar, upC, uppC);
                return firstDouble(JAX.grad(partial).apply(scalarF64(u)));
            }
            @Override
            public double dUp(double x, double u, double up, double upp) {
                final NDArray xC = scalarF64(x), uC = scalarF64(u), uppC = scalarF64(upp);
                Function<NDArray, NDArray> partial = upVar -> fn.apply(xC, uC, upVar, uppC);
                return firstDouble(JAX.grad(partial).apply(scalarF64(up)));
            }
            @Override
            public double dUpp(double x, double u, double up, double upp) {
                final NDArray xC = scalarF64(x), uC = scalarF64(u), upC = scalarF64(up);
                Function<NDArray, NDArray> partial = uppVar -> fn.apply(xC, uC, upC, uppVar);
                return firstDouble(JAX.grad(partial).apply(scalarF64(upp)));
            }
            @Override
            public Batched batched(int size) {
                return new AutodiffBatched(fn, size);
            }
        };
    }

    /**
     * Batched autodiff residual. Each Newton iteration needs three
     * partial derivatives ({@code ∂F/∂u}, {@code ∂F/∂u'}, {@code ∂F/∂u''})
     * evaluated at every grid point. Naively this is
     * {@code 3 * (n+1)} scalar {@code JAX.grad} traces per iteration;
     * batching alone brings it to three vector traces per iteration.
     *
     * <p>This implementation goes a step further: it wraps the four
     * inputs {@code (x, u, u', u'')} in a {@link PyTree} of leaves and
     * runs the entire gradient through {@link JAX#jitGradTree}. The
     * traced Jaxpr is cached at the grid size on first use and reused
     * for every subsequent Newton iteration (and Armijo backtrack) —
     * one trace <em>per grid size</em>, not per iteration. On the
     * grid ladder {@code 16, 32, ..., 512} the whole solve traces
     * at most six times.
     *
     * <p>Correctness relies on the same pointwise assumption as the
     * scalar-loop batched path: {@code F_i} depends only on
     * {@code (x_i, u_i, u'_i, u''_i)}. Then {@code ∂(sum_j F_j)/∂u_i =
     * ∂F_i/∂u_i} so a single scalar-output {@code grad} on the pytree
     * gives us all three diagonals in one shot — the returned pytree's
     * leaves are the vectors of pointwise partials we need.
     */
    private static final class AutodiffBatched implements Batched {
        private final AutodiffFn fn;
        private final Shape shape;
        // Traced once at construction. The Jaxpr's four input Vars are
        // (x, u, up, upp) — no closure-captured constants, so the same
        // Jaxpr correctly evaluates with any new value tuple at the same
        // signature. Cached forever for this Batched instance's lifetime,
        // which itself is bounded to a single grid size.
        private final Jaxpr residualJaxpr;
        private final Jaxpr gradJaxpr;
        AutodiffBatched(AutodiffFn fn, int size) {
            this.fn = fn;
            this.shape = new Shape(size);
            this.residualJaxpr = traceResidual(fn, shape);
            this.gradJaxpr = traceGradOfSum(fn, shape);
        }
        @Override
        public double[] residuals(double[] x, double[] u, double[] up, double[] upp) {
            List<NDArray> outs = Grad.forwardInterpret(residualJaxpr,
                List.of(vec(x), vec(u), vec(up), vec(upp)));
            return outs.get(0).toDoubleArray().clone();
        }
        @Override
        public double[] dU(double[] x, double[] u, double[] up, double[] upp) {
            return allPartials(x, u, up, upp)[0];
        }
        @Override
        public double[] dUp(double[] x, double[] u, double[] up, double[] upp) {
            return allPartials(x, u, up, upp)[1];
        }
        @Override
        public double[] dUpp(double[] x, double[] u, double[] up, double[] upp) {
            return allPartials(x, u, up, upp)[2];
        }
        @Override
        public double[][] allPartials(double[] x, double[] u, double[] up, double[] upp) {
            // Backward-interpret the cached gradient Jaxpr against the
            // current input tuple. This yields four gradient vectors —
            // one per input — and we return the last three (∂/∂u,
            // ∂/∂u', ∂/∂u''). The ∂/∂x gradient is discarded.
            NDArray seed = onesOfShape(shape);
            List<NDArray> grads = Grad.backwardInterpret(gradJaxpr,
                List.of(vec(x), vec(u), vec(up), vec(upp)), List.of(seed));
            return new double[][]{
                grads.get(1).toDoubleArray().clone(),
                grads.get(2).toDoubleArray().clone(),
                grads.get(3).toDoubleArray().clone()
            };
        }
        private NDArray vec(double[] v) {
            return new ConcreteNDArray(v.clone(), shape);
        }
        private static NDArray onesOfShape(Shape shape) {
            // The gradJaxpr's output is a scalar (sum) — a length-1 array.
            return new ConcreteNDArray(new double[]{1.0}, new Shape(1));
        }
        private static Jaxpr traceResidual(AutodiffFn fn, Shape shape) {
            Tracer.start();
            try {
                Var vx   = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                Var vu   = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                Var vup  = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                Var vupp = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                NDArray tx = new TracedNDArray(vx);
                NDArray tu = new TracedNDArray(vu);
                NDArray tup = new TracedNDArray(vup);
                NDArray tupp = new TracedNDArray(vupp);
                NDArray result = fn.apply(tx, tu, tup, tupp);
                Var outVar = ((TracedNDArray) result).getVar();
                return Tracer.stop(List.of(vx, vu, vup, vupp), List.of(outVar));
            } catch (RuntimeException | Error e) {
                Tracer.abort();
                throw e;
            }
        }
        private static Jaxpr traceGradOfSum(AutodiffFn fn, Shape shape) {
            Tracer.start();
            try {
                Var vx   = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                Var vu   = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                Var vup  = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                Var vupp = Tracer.current().nextVar(shape, com.marmanis.jax4j.core.DType.FLOAT64);
                NDArray tx = new TracedNDArray(vx);
                NDArray tu = new TracedNDArray(vu);
                NDArray tup = new TracedNDArray(vup);
                NDArray tupp = new TracedNDArray(vupp);
                NDArray result = fn.apply(tx, tu, tup, tupp).sum();
                Var outVar = ((TracedNDArray) result).getVar();
                return Tracer.stop(List.of(vx, vu, vup, vupp), List.of(outVar));
            } catch (RuntimeException | Error e) {
                Tracer.abort();
                throw e;
            }
        }
    }

    /**
     * Wrap a scalar constant as a length-1 FLOAT64 NDArray. Useful when
     * writing an {@link AutodiffFn} that references a Java {@code double}
     * from the enclosing scope — {@code NDArray} operations don't accept
     * raw doubles, so you write {@code u.exp().mul(scalar(lambda))}.
     */
    public static NDArray scalar(double v) {
        return new ConcreteNDArray(new double[]{v}, new Shape(1));
    }

    private static NDArray scalarF64(double v) {
        return new ConcreteNDArray(new double[]{v}, new Shape(1));
    }

    private static double firstDouble(NDArray a) {
        return a.toDoubleArray()[0];
    }

    private static final int[] GRID_SIZES = {16, 32, 64, 128, 256, 512};
    private static final int MAX_BACKTRACKS = 5;

    private final Domain domain;
    private final Residual residual;

    public NonlinearChebop(Domain domain, Residual residual) {
        this.domain = domain;
        this.residual = residual;
    }

    public Domain domain() { return domain; }

    /**
     * Solve the nonlinear BVP with the given boundary conditions and
     * default Newton options.
     */
    public Chebfun solve(BoundaryCondition bcA, BoundaryCondition bcB) {
        return solve(bcA, bcB, NewtonOptions.defaults());
    }

    public Chebfun solve(BoundaryCondition bcA, BoundaryCondition bcB, NewtonOptions opts) {
        Chebfun last = null;
        for (int n : GRID_SIZES) {
            double[] uVals = solveAt(n, bcA, bcB, opts, last);
            Chebfun candidate = new Chebfun(Chebtech.fromValues(uVals), domain);
            if (last != null) {
                double err = maxAbsDiff(candidate, last);
                double scale = Math.max(1.0, ArrayMath.maxAbs(uVals));
                if (err <= opts.tol() * scale) return candidate;
            }
            last = candidate;
        }
        return last;
    }

    private double[] solveAt(int n, BoundaryCondition bcA, BoundaryCondition bcB,
                             NewtonOptions opts, Chebfun previousSolution) {
        int size = n + 1;
        double[] scaledD = scaledDMatrix(n);
        double[] scaledD2 = DifferentiationMatrix.matMul(scaledD, scaledD, size);
        double[] xGrid = physicalGrid(n);

        // Cache the batched residual view once per grid size. For the
        // autodiff wrapper this is where the vector trace amortization
        // happens: every Newton iteration and every Armijo backtrack goes
        // through the same batched instance rather than allocating fresh
        // scalar closures.
        Batched batched = residual.batched(size);

        // Initial guess: reuse previous grid's solution if available, else
        // fall back to the user-supplied initial guess, else zero.
        double[] u = seedInitialGuess(size, xGrid, opts, previousSolution);

        double[] bcRow0 = readBcRow(bcB, /*row=*/0, /*evalCol=*/0, scaledD, size);
        double[] bcRowN = readBcRow(bcA, /*row=*/n, /*evalCol=*/n, scaledD, size);

        double damping = opts.initialDamping();
        for (int iter = 0; iter < opts.maxIter(); iter++) {
            double[] up  = matVec(scaledD,  u, size);
            double[] upp = matVec(scaledD2, u, size);
            double[] r = batched.residuals(xGrid, u, up, upp);
            r[0] = dot(bcRow0, u) - bcB.value();
            r[n] = dot(bcRowN, u) - bcA.value();

            double resNorm = ArrayMath.maxAbs(r);
            if (resNorm < opts.tol()) return u;

            // Batched Jacobian assembly. One vector call per partial —
            // for the autodiff residual, that's three JAX.grad traces per
            // Newton iteration (down from 3*(n+1) with the old scalar
            // API); for the FD residual it's just three plain scalar
            // loops.
            double[][] partials = batched.allPartials(xGrid, u, up, upp);
            double[] fuVec   = partials[0];
            double[] fupVec  = partials[1];
            double[] fuppVec = partials[2];
            double[] J = new double[size * size];
            for (int i = 1; i <= n - 1; i++) {
                double fu   = fuVec  [i];
                double fup  = fupVec [i];
                double fupp = fuppVec[i];
                for (int j = 0; j < size; j++) {
                    double d1 = scaledD [i * size + j];
                    double d2 = scaledD2[i * size + j];
                    double delta = (i == j) ? 1.0 : 0.0;
                    J[i * size + j] = fu * delta + fup * d1 + fupp * d2;
                }
            }
            for (int j = 0; j < size; j++) {
                J[0 * size + j] = bcRow0[j];
                J[n * size + j] = bcRowN[j];
            }

            NDArray Jarr = new ConcreteNDArray(J, new Shape(size, size));
            double[] negR = new double[size];
            for (int i = 0; i < size; i++) negR[i] = -r[i];
            NDArray rArr = new ConcreteNDArray(negR, new Shape(size));
            double[] du;
            try {
                du = Linalg.solve(Jarr, rArr).toDoubleArray();
            } catch (RuntimeException e) {
                return u;
            }

            // Armijo backtrack — batched residual evaluation only, no
            // Jacobian recomputation.
            double bestNorm = Double.POSITIVE_INFINITY;
            double[] bestU = null;
            double stepSize = damping;
            for (int back = 0; back <= MAX_BACKTRACKS; back++) {
                double[] uTry = new double[size];
                for (int i = 0; i < size; i++) uTry[i] = u[i] + stepSize * du[i];
                double[] upTry  = matVec(scaledD,  uTry, size);
                double[] uppTry = matVec(scaledD2, uTry, size);
                double[] rTry = batched.residuals(xGrid, uTry, upTry, uppTry);
                rTry[0] = dot(bcRow0, uTry) - bcB.value();
                rTry[n] = dot(bcRowN, uTry) - bcA.value();
                double normTry = ArrayMath.maxAbs(rTry);
                if (normTry < bestNorm) { bestNorm = normTry; bestU = uTry; }
                if (normTry < resNorm) break;
                stepSize *= 0.5;
            }
            if (bestU == null) return u;
            u = bestU;

            double duNorm = ArrayMath.maxAbs(du) * stepSize;
            double uScale = Math.max(1.0, ArrayMath.maxAbs(u));
            if (duNorm <= opts.tol() * uScale) return u;
        }
        return u;
    }

    private double[] seedInitialGuess(int size, double[] xGrid,
                                      NewtonOptions opts, Chebfun previousSolution) {
        double[] u = new double[size];
        if (previousSolution != null) {
            for (int i = 0; i < size; i++) u[i] = previousSolution.feval(xGrid[i]);
            return u;
        }
        Chebfun guess = opts.initialGuess();
        if (guess != null) {
            for (int i = 0; i < size; i++) u[i] = guess.feval(xGrid[i]);
        }
        return u;
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    private double[] scaledDMatrix(int n) {
        int size = n + 1;
        double[] D = DifferentiationMatrix.chebD(n);
        double jac = 2.0 / domain.length();
        double[] out = new double[size * size];
        for (int i = 0; i < size * size; i++) out[i] = jac * D[i];
        return out;
    }

    private double[] physicalGrid(int n) {
        double[] out = new double[n + 1];
        for (int j = 0; j <= n; j++) out[j] = domain.fromRef(Math.cos(Math.PI * j / n));
        return out;
    }

    /** Extract the linear-form row a BC contributes at the given (row, evalCol). */
    private static double[] readBcRow(BoundaryCondition bc, int row, int evalCol,
                                      double[] scaledD, int size) {
        double[] rowVec = new double[size];
        int evalOff = evalCol * size;
        switch (bc) {
            case BoundaryCondition.Dirichlet d -> rowVec[evalCol] = 1.0;
            case BoundaryCondition.Neumann n -> {
                for (int j = 0; j < size; j++) rowVec[j] = scaledD[evalOff + j];
            }
            case BoundaryCondition.Robin r -> {
                rowVec[evalCol] += r.alpha();
                for (int j = 0; j < size; j++) rowVec[j] += r.beta() * scaledD[evalOff + j];
            }
        }
        return rowVec;
    }

    private static double[] matVec(double[] mat, double[] v, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) s += mat[i * n + j] * v[j];
            out[i] = s;
        }
        return out;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /**
     * Local variant of {@link ArrayMath#maxAbsDiff(Chebfun, Chebfun)} that
     * samples at {@code max(a.length, b.length) * 2} points rather than
     * the shared helper's {@code min(a.length, b.length) * 4}. When the
     * two iterates differ substantially in resolved length — common in
     * the Newton continuation as grid size doubles — sampling at the
     * <em>larger</em> length picks up detail the coarser iterate resolved
     * away, giving a tighter convergence estimate. The other adaptive
     * solvers ({@link Chebop}, {@link LinearBlockChebop},
     * {@link NonlinearSystem}) compare successive iterates at comparable
     * resolutions and use the min-based helper.
     */
    private static double maxAbsDiff(Chebfun a, Chebfun b) {
        int n = Math.max(a.length(), b.length()) * 2;
        n = Math.max(n, 32);
        double lo = a.domain().a();
        double hi = a.domain().b();
        double dx = (hi - lo) / (n - 1);
        double maxErr = 0.0;
        for (int i = 0; i < n; i++) {
            double x = lo + i * dx;
            double err = Math.abs(a.feval(x) - b.feval(x));
            if (err > maxErr) maxErr = err;
        }
        return maxErr;
    }
}
