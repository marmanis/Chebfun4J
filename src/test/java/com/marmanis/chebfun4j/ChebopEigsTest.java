package com.marmanis.chebfun4j;

import com.marmanis.chebfun4j.BoundaryCondition.Dirichlet;
import com.marmanis.chebfun4j.BoundaryCondition.Neumann;
import com.marmanis.chebfun4j.BoundaryCondition.Robin;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Sturm-Liouville eigenvalue problems solved by Chebop.eigs. Each test
 * picks an ODE + BC combination with a closed-form eigenvalue sequence
 * and verifies the smallest few numerical eigenvalues match analytic
 * ones. Also spot-checks the eigenfunctions are L^2-normalized and
 * satisfy the ODE.
 */
public class ChebopEigsTest {

    private static void assertClose(double expected, double actual, double tol, String msg) {
        if (Math.abs(expected - actual) > tol) {
            throw new AssertionError(msg + ": expected " + expected + " got " + actual);
        }
    }

    @Test
    public void testSchroedingerFreeParticleOnZeroPi() {
        // -u'' = lambda u on [0, pi] with u(0) = u(pi) = 0.
        // Eigenvalues: lambda_n = n^2 for n = 1, 2, 3, ...
        // Written as (-1) * u'' - 0 * u' - lambda * u = 0, i.e.
        // L = -d^2/dx^2 with lambda extracted implicitly. Chebop.eigs
        // solves L u = lambda u, so we make L = -u''.
        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.zero(d).plus(2, -1.0); // L u = -u''
        Chebop.Eigs eigs = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 5);
        double[] want = {1, 4, 9, 16, 25};
        for (int i = 0; i < 5; i++) {
            assertClose(want[i], eigs.eigenvalues()[i], 1e-6, "lambda_" + (i + 1));
        }
        // Eigenfunctions should be L^2-normalized.
        for (int i = 0; i < 5; i++) {
            double n2 = eigs.eigenfunctions().get(i).norm2();
            assertClose(1.0, n2, 1e-8, "||v_" + i + "||_2 = 1");
        }
    }

    @Test
    public void testFirstEigenfunctionShapeIsSin() {
        // First eigenfunction on [0, pi] for -u'' u = lambda u, Dirichlet:
        // v_1(x) proportional to sin(x). Check the ratio at an interior
        // sample is constant (up to sign).
        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.zero(d).plus(2, -1.0);
        Chebop.Eigs eigs = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 1);
        Chebfun v1 = eigs.eigenfunctions().get(0);
        double vAtPi2 = v1.feval(Math.PI / 2);
        double vAtPi4 = v1.feval(Math.PI / 4);
        // Ratio sin(pi/2) / sin(pi/4) = 1 / (1/sqrt(2)) = sqrt(2).
        double ratio = vAtPi2 / vAtPi4;
        assertClose(Math.sqrt(2.0), Math.abs(ratio), 1e-6, "v1 shape ratio");
    }

    @Test
    public void testNeumannEigenvaluesOnZeroPi() {
        // -u'' = lambda u on [0, pi] with u'(0) = u'(pi) = 0.
        // Eigenvalues: lambda_n = n^2, n = 0, 1, 2, ...
        // Eigenfunctions: cos(n x). Note lambda_0 = 0 with the constant
        // eigenfunction.
        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.zero(d).plus(2, -1.0);
        Chebop.Eigs eigs = L.eigs(new Neumann(0.0), new Neumann(0.0), 5);
        // The first eigenvalue should be ~ 0 (constant mode), then 1, 4, 9, 16.
        double[] want = {0, 1, 4, 9, 16};
        for (int i = 0; i < 5; i++) {
            assertClose(want[i], eigs.eigenvalues()[i], 1e-6, "neumann lambda_" + i);
        }
    }

    @Test
    public void testMixedDirichletNeumannEigenvalues() {
        // -u'' = lambda u on [0, pi/2] with u(0) = 0 and u'(pi/2) = 0.
        // Eigenvalues: lambda_n = (2n+1)^2, n = 0, 1, 2, ... (quarter-period modes)
        // Eigenfunctions: sin((2n+1) x).
        Domain d = new Domain(0.0, Math.PI / 2);
        Chebop L = Chebop.zero(d).plus(2, -1.0);
        Chebop.Eigs eigs = L.eigs(new Dirichlet(0.0), new Neumann(0.0), 4);
        double[] want = {1, 9, 25, 49};
        for (int i = 0; i < 4; i++) {
            assertClose(want[i], eigs.eigenvalues()[i], 1e-5, "mixed lambda_" + i);
        }
    }

    @Test
    public void testRobinEigenvalues() {
        // -u'' = lambda u on [0, 1] with u(0) - u'(0) = 0 and u(1) = 0.
        // (The BC combination u(0)+u'(0)=0, u(1)=0 admits the null vector
        // u(x) = c(1-x) for any c: a linear function satisfies both BCs and
        // -u''=0, so lambda=0 is a legitimate eigenvalue. Using
        // u(0)-u'(0)=0 rules out any linear solution and keeps the spectrum
        // purely positive.)
        //
        // Eigenfunctions u = sin(sqrt(lambda) (x - 1)) satisfy u(1)=0.
        // BC u(0)-u'(0) = -sin(k) - k cos(k) = 0 ⟺ tan(k) = -k.
        // Roots of tan(k) = -k for k > 0: k_1 ~ 2.028758, k_2 ~ 4.913180,
        // k_3 ~ 7.978666, so lambda_i ~ 4.116, 24.139, 63.660.
        Domain d = new Domain(0.0, 1.0);
        Chebop L = Chebop.zero(d).plus(2, -1.0);
        Chebop.Eigs eigs = L.eigs(new Robin(1.0, -1.0, 0.0), new Dirichlet(0.0), 3);
        double k1 = 2.028757838110434;
        double k2 = 4.913180439434884;
        double k3 = 7.978665712493970;
        double[] want = {k1 * k1, k2 * k2, k3 * k3};
        for (int i = 0; i < 3; i++) {
            assertClose(want[i], eigs.eigenvalues()[i], 1e-4, "robin lambda_" + i);
        }
    }

    @Test
    public void testFewEigenvaluesConvergeAsNGrows() {
        // Sanity: for the same problem, requesting the first 3
        // eigenvalues should give the same values as requesting the first
        // 5 (they're a subset).
        Domain d = new Domain(0.0, Math.PI);
        Chebop L = Chebop.zero(d).plus(2, -1.0);
        Chebop.Eigs eigs3 = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 3);
        Chebop.Eigs eigs5 = L.eigs(new Dirichlet(0.0), new Dirichlet(0.0), 5);
        for (int i = 0; i < 3; i++) {
            assertClose(eigs3.eigenvalues()[i], eigs5.eigenvalues()[i], 1e-6, "lambda_" + i);
        }
        assertTrue(eigs5.eigenvalues().length >= 5, "requested 5 eigenvalues");
    }
}
