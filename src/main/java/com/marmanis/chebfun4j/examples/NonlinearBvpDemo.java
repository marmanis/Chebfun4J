package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.NewtonOptions;
import com.marmanis.chebfun4j.NonlinearChebop;

/**
 * The classical <a href="https://en.wikipedia.org/wiki/Bratu%E2%80%93Gelfand_problem">
 * Bratu problem</a>: {@code u'' + λ e^u = 0} on {@code [0, 1]} with
 * {@code u(0) = u(1) = 0}. Solved by damped Newton on a spectral
 * collocation grid. For {@code λ < 3.5138...} there are two solutions
 * (upper and lower branch); this demo captures the lower-branch
 * solution for {@code λ = 1}, {@code λ = 2}, and {@code λ = 3}, then
 * prints the max value of {@code u} on the interior.
 */
public class NonlinearBvpDemo {
    public static void main(String[] args) {
        System.out.println("chebfun4j: nonlinear BVP demo (Bratu)");
        System.out.println("=====================================");

        Domain d = new Domain(0.0, 1.0);
        for (double lam : new double[]{1.0, 2.0, 3.0}) {
            NonlinearChebop.Residual F = (x, u, up, upp) -> upp + lam * Math.exp(u);
            NonlinearChebop N = new NonlinearChebop(d, F);
            Chebfun u = N.solve(new Dirichlet(0.0), new Dirichlet(0.0), NewtonOptions.defaults());
            Chebfun.Extremum maxU = u.max();
            System.out.printf("lambda = %.1f:  max u = %.10f at x = %.6f  (chebfun length = %d)%n",
                lam, maxU.value(), maxU.location(), u.length());
        }
        System.out.println();

        // Duffing-like: u'' - u^3 = 0 with u(0)=0, u(1)=1. Same problem, but
        // using the AUTODIFF residual DSL — partials come out of jax4j
        // reverse-mode AD instead of finite differences on the scalar F.
        NonlinearChebop.Residual cubic = NonlinearChebop.autodiffResidual(
            (x, u, up, upp) -> upp.sub(u.mul(u).mul(u)));
        NonlinearChebop Nc = new NonlinearChebop(d, cubic);
        Chebfun uc = Nc.solve(new Dirichlet(0.0), new Dirichlet(1.0));
        System.out.printf("u'' - u^3 = 0 on [0, 1], u(0)=0, u(1)=1  (autodiff residual):%n");
        System.out.printf("  chebfun length = %d%n", uc.length());
        Chebfun d2 = uc.diff().diff();
        double maxResid = 0.0;
        for (double x : new double[]{0.1, 0.25, 0.5, 0.75, 0.9}) {
            double val = uc.feval(x);
            double resid = d2.feval(x) - val * val * val;
            maxResid = Math.max(maxResid, Math.abs(resid));
            System.out.printf("  u(%.2f) = %.10f,  residual = %.2e%n", x, val, resid);
        }
        System.out.printf("  max interior residual: %.2e%n", maxResid);
    }
}
