package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.NewtonOptions;
import com.marmanis.chebfun4j.NonlinearChebop;

/**
 * Wall-clock micro-benchmark comparing the three residual specification
 * paths on the Bratu problem {@code u'' + λ e^u = 0}:
 *
 * <ol>
 *   <li><b>FD</b> — plain scalar {@code Residual} with default centered
 *       finite-difference partials.</li>
 *   <li><b>AD scalar</b> — {@code autodiffResidual} that computes each
 *       partial by a per-grid-point {@code JAX.grad} trace (3×(n+1) traces
 *       per Newton iteration). This is what iteration 4's initial autodiff
 *       wiring did.</li>
 *   <li><b>AD batched</b> — the current {@code autodiffResidual}
 *       {@link com.marmanis.chebfun4j.NonlinearChebop.Batched} override:
 *       one vector {@code JAX.grad} trace per partial per Newton iteration
 *       via the pointwise-sum trick (3 traces per iter regardless of
 *       grid size).</li>
 * </ol>
 *
 * <p>To isolate the residual work from adaptive-grid overhead we solve at
 * a fixed grid size by pre-supplying an initial guess and stepping through
 * the grid ladder manually.
 */
public class NonlinearBench {

    public static void main(String[] args) {
        System.out.println("chebfun4j: nonlinear residual perf benchmark");
        System.out.println("============================================");
        System.out.println();

        // JIT warm-up so the reported numbers reflect steady-state timings.
        for (int i = 0; i < 3; i++) { solve(0, 1.0); solve(1, 1.0); solve(2, 1.0); }

        int repeats = 10;
        double[] lambdas = {1.0, 2.0, 3.0};
        for (double lam : lambdas) {
            System.out.printf("Bratu, λ = %.1f  (u'' + λ e^u = 0, u(0) = u(1) = 0)%n", lam);
            double[] avgs = new double[3];
            for (int mode = 0; mode < 3; mode++) {
                long t0 = System.nanoTime();
                for (int r = 0; r < repeats; r++) solve(mode, lam);
                long t1 = System.nanoTime();
                avgs[mode] = (t1 - t0) / 1e6 / repeats;
            }
            String[] labels = {
                "FD (centered finite differences)",
                "AD scalar (per-grid-point JAX.grad)",
                "AD batched (vector JAX.grad per partial)"
            };
            for (int mode = 0; mode < 3; mode++) {
                System.out.printf("  %-45s avg %.2f ms/solve%n", labels[mode], avgs[mode]);
            }
            System.out.printf("  speedup (batched vs FD):     %.1f×%n", avgs[0] / avgs[2]);
            System.out.printf("  speedup (batched vs scalar): %.1f×%n", avgs[1] / avgs[2]);
            System.out.println();
        }
    }

    private static Chebfun solve(int mode, double lam) {
        Domain d = new Domain(0.0, 1.0);
        NonlinearChebop.Residual F = switch (mode) {
            case 0 -> (x, u, up, upp) -> upp + lam * Math.exp(u);
            case 1 -> scalarAutodiff(lam);
            case 2 -> NonlinearChebop.autodiffResidual(
                (x, u, up, upp) -> upp.add(u.exp().mul(NonlinearChebop.scalar(lam))));
            default -> throw new IllegalStateException();
        };
        NonlinearChebop N = new NonlinearChebop(d, F);
        return N.solve(new Dirichlet(0.0), new Dirichlet(0.0), NewtonOptions.defaults());
    }

    /**
     * A scalar-API autodiff residual: the same math as the "AD batched"
     * variant, but by declining to override {@link
     * com.marmanis.chebfun4j.NonlinearChebop.Residual#batched batched},
     * every partial goes back through the per-grid-point JAX.grad
     * scalar closure. Used to measure the batching win directly.
     */
    private static NonlinearChebop.Residual scalarAutodiff(double lam) {
        NonlinearChebop.Residual delegate = NonlinearChebop.autodiffResidual(
            (x, u, up, upp) -> upp.add(u.exp().mul(NonlinearChebop.scalar(lam))));
        // Anonymous class that hides the batched override.
        return new NonlinearChebop.Residual() {
            @Override public double at(double x, double u, double up, double upp) {
                return delegate.at(x, u, up, upp);
            }
            @Override public double dU(double x, double u, double up, double upp) {
                return delegate.dU(x, u, up, upp);
            }
            @Override public double dUp(double x, double u, double up, double upp) {
                return delegate.dUp(x, u, up, upp);
            }
            @Override public double dUpp(double x, double u, double up, double upp) {
                return delegate.dUpp(x, u, up, upp);
            }
            // Deliberately do NOT override batched(int) — so the scalar
            // loop implementation runs, calling dU / dUp / dUpp per grid
            // point.
        };
    }
}
