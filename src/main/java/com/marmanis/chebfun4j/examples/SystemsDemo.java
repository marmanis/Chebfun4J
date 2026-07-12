package com.marmanis.chebfun4j.examples;

import com.marmanis.chebfun4j.Chebfun;
import com.marmanis.chebfun4j.Domain;
import com.marmanis.chebfun4j.LinearBlockChebop;
import com.marmanis.chebfun4j.NonlinearSystem;
import com.marmanis.chebfun4j.SystemBC;
import com.marmanis.chebfun4j.SystemBC.Dirichlet;

/**
 * Systems of ODE BVPs: a linear coupled 2-D system solved with
 * {@link LinearBlockChebop}, and a nonlinear coupled system solved with
 * {@link NonlinearSystem}. Both print pointwise values and verify the
 * analytic solution / residual.
 */
public class SystemsDemo {

    public static void main(String[] args) {
        System.out.println("chebfun4j: systems of ODE demos");
        System.out.println("===============================");
        System.out.println();

        // ------------------------------------------------------------
        // Linear coupled system: convert u'' + u = 0 into 1st-order.
        // u_0' = u_1
        // u_1' = -u_0
        // On [0, pi/2] with u_0(0) = 0, u_0(pi/2) = 1.
        // Solution: u_0 = sin(x), u_1 = cos(x).
        // ------------------------------------------------------------
        Domain d = new Domain(0.0, Math.PI / 2);
        LinearBlockChebop L = LinearBlockChebop.zero(d, 2)
            .term(0, 0, 1, 1.0).term(0, 1, 0, -1.0)   // u_0' - u_1 = 0
            .term(1, 0, 0, 1.0).term(1, 1, 1, 1.0);   // u_0 + u_1' = 0
        Chebfun[] rhs = { Chebfun.constant(0.0, d), Chebfun.constant(0.0, d) };
        Chebfun[] u = L.solve(rhs, new SystemBC[]{ new Dirichlet(0, 0.0) },
                                   new SystemBC[]{ new Dirichlet(0, 1.0) });
        System.out.printf("Linear coupled system (u'' + u = 0 as [u_0', u_1'] = [u_1, -u_0]):%n");
        System.out.printf("  lengths          = [%d, %d]%n", u[0].length(), u[1].length());
        for (double x : new double[]{0.2, 0.5, 1.0, 1.3}) {
            System.out.printf("  u_0(%.2f) = %.15f  (sin(x) = %.15f)%n", x, u[0].feval(x), Math.sin(x));
            System.out.printf("  u_1(%.2f) = %.15f  (cos(x) = %.15f)%n", x, u[1].feval(x), Math.cos(x));
        }
        System.out.println();

        // ------------------------------------------------------------
        // Symmetric nonlinear system:
        //   u_0'' + u_1^2 = 0
        //   u_1'' + u_0^2 = 0
        // on [0, 1] with u_0(0) = u_1(0) = 0 and u_0(1) = u_1(1) = 1.
        // By symmetry u_0 == u_1 at the solution.
        // ------------------------------------------------------------
        Domain d2 = new Domain(0.0, 1.0);
        NonlinearSystem.Residual F = new NonlinearSystem.Residual() {
            @Override public int numComponents() { return 2; }
            @Override public double[] at(double x, double[] uu, double[] up, double[] upp) {
                return new double[]{ upp[0] + uu[1] * uu[1], upp[1] + uu[0] * uu[0] };
            }
        };
        NonlinearSystem sys = new NonlinearSystem(d2, F);
        Chebfun[] w = sys.solve(
            new SystemBC[]{ new Dirichlet(0, 0.0), new Dirichlet(1, 0.0) },
            new SystemBC[]{ new Dirichlet(0, 1.0), new Dirichlet(1, 1.0) });
        System.out.printf("Nonlinear system (u_0'' + u_1^2 = 0, u_1'' + u_0^2 = 0):%n");
        System.out.printf("  lengths          = [%d, %d]%n", w[0].length(), w[1].length());
        for (double x : new double[]{0.25, 0.5, 0.75}) {
            System.out.printf("  u_0(%.2f) = %.15f, u_1(%.2f) = %.15f%n",
                              x, w[0].feval(x), x, w[1].feval(x));
        }
        // Residual check.
        Chebfun r0 = w[0].diff().diff().plus(w[1].times(w[1]));
        double maxR = Math.max(r0.normInf(), w[1].diff().diff().plus(w[0].times(w[0])).normInf());
        System.out.printf("  max interior residual: %.3e%n", maxR);
    }
}
