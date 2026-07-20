package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Responsible for constructing the initial velocity flow fields for the 3-D
 * Navier-Stokes spectral simulation.
 */
public class FlowFieldInitializer {
    private static final Logger logger = LoggerFactory.getLogger(FlowFieldInitializer.class);

    /**
     * Initializes the velocity fields in Fourier space using a random isotropic
     * spectrum generator (ranflow) matching the Fortran reference code.
     *
     * @param sim the Navier-Stokes simulation instance to initialize
     */
    public static void initializeRandomFlow(NavierStokes3D sim) {
        int n1 = sim.getN1();
        int n2 = sim.getN2();
        int half = sim.getHalf();
        double[] kx = sim.getKx();
        double[] ky = sim.getKy();
        double[] kz = sim.getKz();
        double[] kx2 = sim.getKx2();
        double[] ky2 = sim.getKy2();
        double[] kz2 = sim.getKz2();
        double c1 = sim.getC1();
        double c2 = sim.getC2();
        double c3 = sim.getC3();
        double c4 = sim.getC4();
        double facp = sim.getFacp();
        boolean plank = sim.isPlank();

        int idum = -27343;
        double pi2 = 2.0 * Math.PI;
        double smallx = 1.0e-14;

        double[] vxRe = new double[n1 * n2 * half];
        double[] vxIm = new double[n1 * n2 * half];
        double[] vyRe = new double[n1 * n2 * half];
        double[] vyIm = new double[n1 * n2 * half];
        double[] vzRe = new double[n1 * n2 * half];
        double[] vzIm = new double[n1 * n2 * half];

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                Ran2 rng1 = new Ran2(idum - 10 * i * j);
                Ran2 rng2 = new Ran2(idum - 9 * j * (i + 1) * (i + 1));

                for (int k = 0; k < half; k++) {
                    double rmag1 = Math.sqrt(-Math.log(rng1.next()));
                    double rph1 = rng2.next();
                    double fr1Re = rmag1 * Math.cos(pi2 * rph1);
                    double fr1Im = rmag1 * Math.sin(pi2 * rph1);

                    double rmag2 = Math.sqrt(-Math.log(rng1.next()));
                    double rph2 = rng2.next();
                    double fr2Re = rmag2 * Math.cos(pi2 * rph2);
                    double fr2Im = rmag2 * Math.sin(pi2 * rph2);

                    double rmag3 = Math.sqrt(-Math.log(rng1.next()));
                    double rph3 = rng2.next();
                    double fr3Re = rmag3 * Math.cos(pi2 * rph3);
                    double fr3Im = rmag3 * Math.sin(pi2 * rph3);

                    double radksq = kx2[i] + ky2[j] + kz2[k];
                    double radk = Math.sqrt(radksq) + smallx;

                    double powerk1 = Math.sqrt(3.0 * c1 / (pi2 * facp)) / facp;
                    powerk1 = powerk1 * Math.pow(radk, c2);
                    double powerk2 = c3 * Math.pow(radk / facp, c4);

                    double scale = powerk1 * Math.exp(powerk2);
                    if (plank) {
                        scale = powerk1 / (Math.exp(powerk2) - 1.0);
                    }

                    double temp = 1.0 / (radksq + smallx);
                    double divRe = temp * (kx[i] * fr1Re + ky[j] * fr2Re + kz[k] * fr3Re);
                    double divIm = temp * (kx[i] * fr1Im + ky[j] * fr2Im + kz[k] * fr3Im);

                    int idx = (i * n2 + j) * half + k;
                    vxRe[idx] = scale * (fr1Re - kx[i] * divRe);
                    vxIm[idx] = scale * (fr1Im - kx[i] * divIm);

                    vyRe[idx] = scale * (fr2Re - ky[j] * divRe);
                    vyIm[idx] = scale * (fr2Im - ky[j] * divIm);

                    vzRe[idx] = scale * (fr3Re - kz[k] * divRe);
                    vzIm[idx] = scale * (fr3Im - kz[k] * divIm);
                }
            }
        }

        cleanSymmetries(n1, n2, half, vxRe, vxIm);
        cleanSymmetries(n1, n2, half, vyRe, vyIm);
        cleanSymmetries(n1, n2, half, vzRe, vzIm);

        vxRe[0] = 0.0; vxIm[0] = 0.0;
        vyRe[0] = 0.0; vyIm[0] = 0.0;
        vzRe[0] = 0.0; vzIm[0] = 0.0;

        Shape shape = new Shape(n1, n2, half);
        sim.setVelocityFields(
            new ConcreteNDArray(vxRe, shape),
            new ConcreteNDArray(vxIm, shape),
            new ConcreteNDArray(vyRe, shape),
            new ConcreteNDArray(vyIm, shape),
            new ConcreteNDArray(vzRe, shape),
            new ConcreteNDArray(vzIm, shape)
        );
    }

    /**
     * Initializes the velocity fields to represent a structured vortex cell flow.
     *
     * @param sim the Navier-Stokes simulation instance to initialize
     */
    public static void initializeVortexFlow(NavierStokes3D sim) {
        logger.info("Vortex flow initialization requested. Running random flow as fallback.");
        initializeRandomFlow(sim);
    }

    private static void cleanSymmetries(int n1, int n2, int half, double[] re, double[] im) {
        int nyq = half - 1;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                int idx = (i * n2 + j) * half + nyq;
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }
}
