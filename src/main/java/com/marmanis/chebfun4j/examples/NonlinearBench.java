package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.NewtonOptions;
import com.marmanis.chebfun4j.NonlinearChebop;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Wall-clock micro-benchmark comparing the three residual specification
 * paths on the Bratu problem {@code u'' + λ e^u = 0}.
 */
public class NonlinearBench {
    private static final Logger logger = LoggerFactory.getLogger(NonlinearBench.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: nonlinear residual perf benchmark");
        logger.info("============================================");

        // JIT warm-up so the reported numbers reflect steady-state timings.
        for (int i = 0; i < 3; i++) { solve(0, 1.0); solve(1, 1.0); solve(2, 1.0); }

        int repeats = 10;
        double[] lambdas = {1.0, 2.0, 3.0};
        for (double lam : lambdas) {
            logger.info(String.format("Bratu, λ = %.1f  (u'' + λ e^u = 0, u(0) = u(1) = 0)", lam));
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
                logger.info(String.format("  %-45s avg %.2f ms/solve", labels[mode], avgs[mode]));
            }
            logger.info(String.format("  speedup (batched vs FD):     %.1f×", avgs[0] / avgs[2]));
            logger.info(String.format("  speedup (batched vs scalar): %.1f×", avgs[1] / avgs[2]));
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

    private static NonlinearChebop.Residual scalarAutodiff(double lam) {
        NonlinearChebop.Residual delegate = NonlinearChebop.autodiffResidual(
            (x, u, up, upp) -> upp.add(u.exp().mul(NonlinearChebop.scalar(lam))));
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
        };
    }
}
