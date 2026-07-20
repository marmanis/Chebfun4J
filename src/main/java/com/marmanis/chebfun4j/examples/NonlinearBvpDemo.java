package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.NewtonOptions;
import com.marmanis.chebfun4j.NonlinearChebop;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The classical Bratu problem: {@code u'' + λ e^u = 0} on {@code [0, 1]} with
 * {@code u(0) = u(1) = 0}.
 */
public class NonlinearBvpDemo {
    private static final Logger logger = LoggerFactory.getLogger(NonlinearBvpDemo.class);

    public static void main(String[] args) {
        logger.info("chebfun4j: nonlinear BVP demo (Bratu)");
        logger.info("=====================================");

        Domain d = new Domain(0.0, 1.0);
        for (double lam : new double[]{1.0, 2.0, 3.0}) {
            NonlinearChebop.Residual F = (x, u, up, upp) -> upp + lam * Math.exp(u);
            NonlinearChebop N = new NonlinearChebop(d, F);
            Chebfun u = N.solve(new Dirichlet(0.0), new Dirichlet(0.0), NewtonOptions.defaults());
            Chebfun.Extremum maxU = u.max();
            logger.info(String.format("lambda = %.1f:  max u = %.10f at x = %.6f  (chebfun length = %d)",
                lam, maxU.value(), maxU.location(), u.length()));
        }

        NonlinearChebop.Residual cubic = NonlinearChebop.autodiffResidual(
            (x, u, up, upp) -> upp.sub(u.mul(u).mul(u)));
        NonlinearChebop Nc = new NonlinearChebop(d, cubic);
        Chebfun uc = Nc.solve(new Dirichlet(0.0), new Dirichlet(1.0));
        logger.info("u'' - u^3 = 0 on [0, 1], u(0)=0, u(1)=1  (autodiff residual):");
        logger.info("  chebfun length = {}", uc.length());
        Chebfun d2 = uc.diff().diff();
        double maxResid = 0.0;
        for (double x : new double[]{0.1, 0.25, 0.5, 0.75, 0.9}) {
            double val = uc.feval(x);
            double resid = d2.feval(x) - val * val * val;
            maxResid = Math.max(maxResid, Math.abs(resid));
            logger.info(String.format("  u(%.2f) = %.10f,  residual = %.2e", x, val, resid));
        }
        logger.info(String.format("  max interior residual: %.2e", maxResid));
    }
}
