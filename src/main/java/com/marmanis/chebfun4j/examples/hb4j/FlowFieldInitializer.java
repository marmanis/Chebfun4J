package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.api.Random;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.PRNGKey;
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
        int half = sim.getHalf(); // The conjugate-symmetric Fourier half-dimension

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

        double pi2 = 2.0 * Math.PI;
        double smallx = 1.0e-14;

        double[] vxRe = new double[n1 * n2 * half];
        double[] vxIm = new double[n1 * n2 * half];
        double[] vyRe = new double[n1 * n2 * half];
        double[] vyIm = new double[n1 * n2 * half];
        double[] vzRe = new double[n1 * n2 * half];
        double[] vzIm = new double[n1 * n2 * half];

        // Draw the six complex-Gaussian component fields (fr1..fr3 real
        // and imaginary parts) from six INDEPENDENT streams derived from
        // a single root key via jax4j's splittable RNG. Compared with
        // the previous Ran2 setup — which seeded new RNGs per (i, j)
        // and then drew all three velocity components from consecutive
        // next() calls — this gives x/y/z components RNG streams that
        // don't share any state, removing the short-range correlations
        // that were tilting a1/a2/a3 away from the isotropic 1/3 each.
        PRNGKey root = PRNGKey.key(sim.getRandomSeed());
        PRNGKey[] sub = Random.split(root, 6);
        Shape s = new Shape(n1, n2, half);
        float[] fr1Re = Random.normal(sub[0], s).toFloatArray();
        float[] fr1Im = Random.normal(sub[1], s).toFloatArray();
        float[] fr2Re = Random.normal(sub[2], s).toFloatArray();
        float[] fr2Im = Random.normal(sub[3], s).toFloatArray();
        float[] fr3Re = Random.normal(sub[4], s).toFloatArray();
        float[] fr3Im = Random.normal(sub[5], s).toFloatArray();

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < half; k++) {
                    int idx = (i * n2 + j) * half + k;

                    double f1Re = fr1Re[idx], f1Im = fr1Im[idx];
                    double f2Re = fr2Re[idx], f2Im = fr2Im[idx];
                    double f3Re = fr3Re[idx], f3Im = fr3Im[idx];

                    double radksq = kx2[i] + ky2[j] + kz2[k];
                    double radk = Math.sqrt(radksq) + smallx;

                    double powerk1 = Math.sqrt(3.0 * c1 / (pi2 * facp)) / facp;
                    powerk1 = powerk1 * Math.pow(radk, c2);
                    double powerk2 = c3 * Math.pow(radk / facp, c4);

                    double scale;
                    if (plank) {
                        scale = powerk1 / (Math.exp(powerk2) - 1.0);
                    } else {
                        scale = powerk1 * Math.exp(powerk2);
                    }

                    double temp = 1.0 / (radksq + smallx);
                    double divRe = temp * (kx[i] * f1Re + ky[j] * f2Re + kz[k] * f3Re);
                    double divIm = temp * (kx[i] * f1Im + ky[j] * f2Im + kz[k] * f3Im);

                    vxRe[idx] = scale * (f1Re - kx[i] * divRe);
                    vxIm[idx] = scale * (f1Im - kx[i] * divIm);

                    vyRe[idx] = scale * (f2Re - ky[j] * divRe);
                    vyIm[idx] = scale * (f2Im - ky[j] * divIm);

                    vzRe[idx] = scale * (f3Re - kz[k] * divRe);
                    vzIm[idx] = scale * (f3Im - kz[k] * divIm);
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
     * Initializes the velocity field with a SINGLE Burgers-like vortex
     * tube aligned with the z-axis, centred on the box middle
     * {@code (π, π)} in the {@code (x, y)} plane.
     *
     * <p>Uses the same three physical parameters as
     * {@link #initializeVortexFlow} — {@code vAlpha}, {@code vBeta},
     * {@code vortexStrength} — with the same profile shapes:
     * <pre>
     *     u_r = -(vAlpha/2) · r
     *     u_θ =  (Γ / (2π·r)) · (1 − exp(vAlpha·r² / (4·ν)))    (Γ = vortexStrength)
     *     u_z =  vAlpha · z_off · exp(-vBeta · z_off²)
     * </pre>
     * where {@code r} is the perpendicular distance from the tube's
     * (z-axis) centreline and {@code z_off} is the axial distance from
     * the box mid-plane. The Cartesian components use the textbook
     * cylindrical→Cartesian rotation
     * {@code (vₓ, v_y) = (cosθ·u_r − sinθ·u_θ, sinθ·u_r + cosθ·u_θ)}.
     *
     * <p>Post-processing (zero boundary planes → FFT → divergence-free
     * projection → zero DC) mirrors {@link #initializeVortexFlow},
     * so the vRatio / vCell / vStep tiling parameters are unused for
     * this mode.
     */
    public static void initializeVortexTube(NavierStokes3D sim) {
        int n1 = sim.getN1();
        int n2 = sim.getN2();
        int n3 = sim.getN3();
        int half = sim.getHalf();

        double alpha = sim.getVAlpha();
        double beta  = sim.getVBeta();
        double gamma = sim.getVortexStrength();
        double visc  = sim.getRnu();

        double pi2 = 2.0 * Math.PI;
        double smallx = 1.0e-14;
        double dx = pi2 / n2;
        double aOver4nu = alpha / (4.0 * visc);

        // Tube axis passes through the geometric centre of the box in
        // the (x, y) plane. Same convention as vortex_flow uses per-cell.
        double x0 = 0.5 * n1 * dx;
        double y0 = 0.5 * n2 * dx;
        double z0 = 0.5 * n3 * dx;

        logger.info("Vortex tube along z: centre at (π,π), α={} β={} Γ={} ν={}",
            alpha, beta, gamma, visc);

        double[] u = new double[n1 * n2 * n3];
        double[] v = new double[n1 * n2 * n3];
        double[] w = new double[n1 * n2 * n3];

        for (int i = 0; i < n1; i++) {
            double dxOff = i * dx - x0;
            for (int j = 0; j < n2; j++) {
                double dyOff = j * dx - y0;
                double r = Math.sqrt(dxOff * dxOff + dyOff * dyOff) + smallx;
                double r2 = r * r;
                double ur = -0.5 * alpha * r;
                double uth = (gamma / (pi2 * r)) * (1.0 - Math.exp(aOver4nu * r2));
                double costh = dxOff / r;
                double sinth = dyOff / r;
                double ux = costh * ur - sinth * uth;
                double uy = sinth * ur + costh * uth;
                for (int k = 0; k < n3; k++) {
                    double zOff = k * dx - z0;
                    double uz = alpha * zOff * Math.exp(-beta * zOff * zOff);
                    int idx = (i * n2 + j) * n3 + k;
                    u[idx] = ux;
                    v[idx] = uy;
                    w[idx] = uz;
                }
            }
        }

        // Same periodicity + spectral cleanup as vortex_flow.
        zeroPlanes(u, v, w, n1, n2, n3);

        Shape realShape = new Shape(n1, n2, n3);
        NDArray[] uSpec = Fft.rfft3(new ConcreteNDArray(u, realShape));
        NDArray[] vSpec = Fft.rfft3(new ConcreteNDArray(v, realShape));
        NDArray[] wSpec = Fft.rfft3(new ConcreteNDArray(w, realShape));

        double[] vxRe = uSpec[0].toDoubleArray();
        double[] vxIm = uSpec[1].toDoubleArray();
        double[] vyRe = vSpec[0].toDoubleArray();
        double[] vyIm = vSpec[1].toDoubleArray();
        double[] vzRe = wSpec[0].toDoubleArray();
        double[] vzIm = wSpec[1].toDoubleArray();

        double[] kx = sim.getKx();
        double[] ky = sim.getKy();
        double[] kz = sim.getKz();
        double[] kx2 = sim.getKx2();
        double[] ky2 = sim.getKy2();
        double[] kz2 = sim.getKz2();
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < half; k++) {
                    int idx = (i * n2 + j) * half + k;
                    double radksq = kx2[i] + ky2[j] + kz2[k];
                    double temp = 1.0 / (radksq + smallx);
                    double divRe = temp * (kx[i]*vxRe[idx] + ky[j]*vyRe[idx] + kz[k]*vzRe[idx]);
                    double divIm = temp * (kx[i]*vxIm[idx] + ky[j]*vyIm[idx] + kz[k]*vzIm[idx]);
                    vxRe[idx] -= kx[i] * divRe;   vxIm[idx] -= kx[i] * divIm;
                    vyRe[idx] -= ky[j] * divRe;   vyIm[idx] -= ky[j] * divIm;
                    vzRe[idx] -= kz[k] * divRe;   vzIm[idx] -= kz[k] * divIm;
                }
            }
        }

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
     * Initializes the velocity field with TWO parallel Burgers-like
     * vortex tubes, both aligned with the z-axis, separated along x by
     * {@code sim.getVortexTubeSeparation()}.
     *
     * <p>Tube centres in the (x, y) plane:
     * <pre>
     *     tube 0: (π − sep/2, π)
     *     tube 1: (π + sep/2, π)
     * </pre>
     *
     * <p>Each tube uses its OWN entries from the array parameters
     * {@code vAlpha[i]}, {@code vBeta[i]}, {@code vortexStrength[i]}
     * (i = 0, 1). The sign of {@code vortexStrength[i]} controls the
     * rotation sense of that tube — use opposite signs for a counter-
     * rotating vortex dipole, same signs for a co-rotating pair.
     *
     * <p>The two velocity fields are summed pointwise before the
     * standard periodicity + divergence-free projection cleanup, so the
     * result is a genuinely superposed pair rather than a mask.
     */
    public static void initializeTwoVortexTubes(NavierStokes3D sim) {
        int n1 = sim.getN1();
        int n2 = sim.getN2();
        int n3 = sim.getN3();
        int half = sim.getHalf();

        double[] alphas = sim.getVAlphaArray();
        double[] betas  = sim.getVBetaArray();
        double[] gammas = sim.getVortexStrengthArray();
        if (alphas.length < 2 || betas.length < 2 || gammas.length < 2) {
            throw new IllegalStateException(
                "initialFlow=4 requires vAlpha, vBeta, vortexStrength to be "
                + "comma-separated arrays with at least 2 entries each. "
                + "Got lengths " + alphas.length + "/" + betas.length + "/" + gammas.length);
        }
        double sep = sim.getVortexTubeSeparation();
        double visc = sim.getRnu();

        double pi2 = 2.0 * Math.PI;
        double smallx = 1.0e-14;
        double dx = pi2 / n2;
        double x0 = 0.5 * n1 * dx;
        double y0 = 0.5 * n2 * dx;
        double z0 = 0.5 * n3 * dx;

        double[] xc = { x0 - 0.5 * sep, x0 + 0.5 * sep };
        double[] yc = { y0,             y0             };

        logger.info("Two vortex tubes along z: separation={}, "
                    + "α={} β={} Γ={} ν={}",
            sep,
            java.util.Arrays.toString(alphas),
            java.util.Arrays.toString(betas),
            java.util.Arrays.toString(gammas), visc);

        double[] u = new double[n1 * n2 * n3];
        double[] v = new double[n1 * n2 * n3];
        double[] w = new double[n1 * n2 * n3];

        for (int t = 0; t < 2; t++) {
            double alpha = alphas[t];
            double beta  = betas[t];
            double gamma = gammas[t];
            double aOver4nu = alpha / (4.0 * visc);
            double cx = xc[t];
            double cy = yc[t];

            for (int i = 0; i < n1; i++) {
                double dxOff = i * dx - cx;
                for (int j = 0; j < n2; j++) {
                    double dyOff = j * dx - cy;
                    double r = Math.sqrt(dxOff * dxOff + dyOff * dyOff) + smallx;
                    double r2 = r * r;
                    double ur = -0.5 * alpha * r;
                    double uth = (gamma / (pi2 * r)) * (1.0 - Math.exp(aOver4nu * r2));
                    double costh = dxOff / r;
                    double sinth = dyOff / r;
                    double ux = costh * ur - sinth * uth;
                    double uy = sinth * ur + costh * uth;
                    for (int k = 0; k < n3; k++) {
                        double zOff = k * dx - z0;
                        double uz = alpha * zOff * Math.exp(-beta * zOff * zOff);
                        int idx = (i * n2 + j) * n3 + k;
                        u[idx] += ux;
                        v[idx] += uy;
                        w[idx] += uz;
                    }
                }
            }
        }

        zeroPlanes(u, v, w, n1, n2, n3);

        Shape realShape = new Shape(n1, n2, n3);
        NDArray[] uSpec = Fft.rfft3(new ConcreteNDArray(u, realShape));
        NDArray[] vSpec = Fft.rfft3(new ConcreteNDArray(v, realShape));
        NDArray[] wSpec = Fft.rfft3(new ConcreteNDArray(w, realShape));

        double[] vxRe = uSpec[0].toDoubleArray();
        double[] vxIm = uSpec[1].toDoubleArray();
        double[] vyRe = vSpec[0].toDoubleArray();
        double[] vyIm = vSpec[1].toDoubleArray();
        double[] vzRe = wSpec[0].toDoubleArray();
        double[] vzIm = wSpec[1].toDoubleArray();

        double[] kx = sim.getKx();
        double[] ky = sim.getKy();
        double[] kz = sim.getKz();
        double[] kx2 = sim.getKx2();
        double[] ky2 = sim.getKy2();
        double[] kz2 = sim.getKz2();
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < half; k++) {
                    int idx = (i * n2 + j) * half + k;
                    double radksq = kx2[i] + ky2[j] + kz2[k];
                    double temp = 1.0 / (radksq + smallx);
                    double divRe = temp * (kx[i]*vxRe[idx] + ky[j]*vyRe[idx] + kz[k]*vzRe[idx]);
                    double divIm = temp * (kx[i]*vxIm[idx] + ky[j]*vyIm[idx] + kz[k]*vzIm[idx]);
                    vxRe[idx] -= kx[i] * divRe;   vxIm[idx] -= kx[i] * divIm;
                    vyRe[idx] -= ky[j] * divRe;   vyIm[idx] -= ky[j] * divIm;
                    vzRe[idx] -= kz[k] * divRe;   vzIm[idx] -= kz[k] * divIm;
                }
            }
        }

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
     * Initializes the velocity fields with a lattice of Burgers-like
     * vortex cells at randomly-oriented axes, mirroring the Fortran
     * {@code vortex_flow} in {@code HB_lib1.f90}.
     *
     * <p>Each cell of half-width {@code vCell} contributes a superposition of
     * a radial sink ({@code u_r = -½ α r}), an azimuthal Lamb-Oseen-like
     * component ({@code u_θ = γ/(2πr)·(1 - exp(αr²/4ν))}), and an axial
     * Gaussian core ({@code u_z = α z_ax exp(-β z_ax²)}). The whole
     * box is tiled with cells of size {@code 2·vCell - 1}; on each
     * cell the axis cycles through x / y / z. After all cells are laid
     * down, {@code vCell} grows by {@code vStep} and a coarser tiling
     * is added on top, repeating until {@code vCell ≥ vRatio·n1}.
     *
     * <p>The result is then forced periodic (boundary planes zeroed),
     * FFT'd to spectral space, projected onto the divergence-free
     * subspace ({@code v̂ ← v̂ − k(k·v̂)/|k|²}), and the DC mode
     * zeroed — same post-processing as {@link #initializeRandomFlow}.
     */
    public static void initializeVortexFlow(NavierStokes3D sim) {
        int n1 = sim.getN1();
        int n2 = sim.getN2();
        int n3 = sim.getN3();
        int half = sim.getHalf();

        double alpha  = sim.getVAlpha();
        double beta   = sim.getVBeta();
        double gamma  = sim.getVortexStrength();
        double visc   = sim.getRnu();
        int vortCell  = sim.getVCell();
        int vortStep  = sim.getVStep();
        double vRatio = sim.getVRatio();

        double pi2 = 2.0 * Math.PI;
        double smallx = 1.0e-14;
        double dx = pi2 / n2;
        double aOver4nu = alpha / (4.0 * visc);

        int maxVortCell = (int) (vRatio * n1);
        logger.info("Vortex flow: maxCell={} (vRatio={}·n1), start vCell={} step={}, α={} β={} γ={} ν={}",
            maxVortCell, vRatio, vortCell, vortStep, alpha, beta, gamma, visc);

        int N = n1 * n2 * n3;
        double[] u = new double[N];   // physical-space x-velocity
        double[] v = new double[N];   // physical-space y-velocity
        double[] w = new double[N];   // physical-space z-velocity

        // Outer while: successively coarser tilings stacked additively.
        // Inner loops use Fortran 1-based indexing translated to 0-based
        // when writing into u/v/w. counter cycles the vortex-axis
        // orientation across boxes so the sample is more isotropic.
        int counter = 0;
        while (vortCell < maxVortCell) {
            int vortBoxStep = 2 * vortCell - 1;
            int boxes = 0;
            for (int i = vortCell; i <= n1 - vortCell; i += vortBoxStep) {
                for (int j = vortCell; j <= n2 - vortCell; j += vortBoxStep) {
                    for (int k = vortCell; k <= n3 - vortCell; k += vortBoxStep) {
                        int orient = counter % 3 + 1;
                        addVortexBox(u, v, w, n2, n3,
                                     i, j, k, vortCell, vortBoxStep,
                                     dx, alpha, beta, gamma, aOver4nu, pi2, smallx,
                                     orient);
                        counter++;
                        boxes++;
                    }
                }
            }
            logger.info("  vCell={} → laid down {} boxes", vortCell, boxes);
            vortCell += vortStep;
        }

        // Force periodicity: zero out the six boundary planes so the
        // subsequent FFT doesn't hallucinate discontinuities that
        // aren't part of the vortex field.
        zeroPlanes(u, v, w, n1, n2, n3);

        // FFT physical → spectral, apply div-free projection, zero DC.
        // Same postprocessing sequence as ranflow.
        Shape realShape = new Shape(n1, n2, n3);
        NDArray[] uSpec = Fft.rfft3(new ConcreteNDArray(u, realShape));
        NDArray[] vSpec = Fft.rfft3(new ConcreteNDArray(v, realShape));
        NDArray[] wSpec = Fft.rfft3(new ConcreteNDArray(w, realShape));

        double[] vxRe = uSpec[0].toDoubleArray();
        double[] vxIm = uSpec[1].toDoubleArray();
        double[] vyRe = vSpec[0].toDoubleArray();
        double[] vyIm = vSpec[1].toDoubleArray();
        double[] vzRe = wSpec[0].toDoubleArray();
        double[] vzIm = wSpec[1].toDoubleArray();

        double[] kx = sim.getKx();
        double[] ky = sim.getKy();
        double[] kz = sim.getKz();
        double[] kx2 = sim.getKx2();
        double[] ky2 = sim.getKy2();
        double[] kz2 = sim.getKz2();
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < half; k++) {
                    int idx = (i * n2 + j) * half + k;
                    double radksq = kx2[i] + ky2[j] + kz2[k];
                    double temp = 1.0 / (radksq + smallx);
                    double divRe = temp * (kx[i]*vxRe[idx] + ky[j]*vyRe[idx] + kz[k]*vzRe[idx]);
                    double divIm = temp * (kx[i]*vxIm[idx] + ky[j]*vyIm[idx] + kz[k]*vzIm[idx]);
                    vxRe[idx] -= kx[i] * divRe;   vxIm[idx] -= kx[i] * divIm;
                    vyRe[idx] -= ky[j] * divRe;   vyIm[idx] -= ky[j] * divIm;
                    vzRe[idx] -= kz[k] * divRe;   vzIm[idx] -= kz[k] * divIm;
                }
            }
        }

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
     * Adds one vortex cell centred on grid indices (i, j, k) (Fortran
     * 1-based) with the given axis orientation into the physical-space
     * velocity arrays. Orientation 1/2/3 = axis along x/y/z; the
     * Fortran source cycles these across successive boxes to encourage
     * isotropy of the aggregate field.
     *
     * <p>Uses the standard cylindrical-to-Cartesian rotation
     * {@code (v_a, v_b) = (cosθ·u_r − sinθ·u_θ, sinθ·u_r + cosθ·u_θ)}
     * for the two radial-plane components of each orientation.
     */
    private static void addVortexBox(double[] u, double[] v, double[] w,
                                     int n2, int n3,
                                     int i, int j, int k,
                                     int vortCell, int vortBoxStep,
                                     double dx, double alpha, double beta, double gamma,
                                     double aOver4nu, double pi2, double smallx,
                                     int orient) {
        for (int a = 1; a <= vortBoxStep; a++) {
            double axial = (a - vortCell) * dx;
            double uz = alpha * axial * Math.exp(-beta * axial * axial);
            for (int b = 1; b <= vortBoxStep; b++) {
                for (int c = 1; c <= vortBoxStep; c++) {
                    int iloc, jloc, kloc;
                    double rA, rB;   // the two radial (perpendicular) offsets
                    switch (orient) {
                        case 1:   // axis = x, radial plane = (y, z)
                            iloc = i + (a - vortCell);
                            jloc = j + (b - vortCell);
                            kloc = k + (c - vortCell);
                            rA = (b - vortCell) * dx;   // y offset → costh numerator
                            rB = (c - vortCell) * dx;   // z offset → sinth numerator
                            break;
                        case 2:   // axis = y, radial plane = (z, x)
                            iloc = i + (b - vortCell);
                            jloc = j + (a - vortCell);
                            kloc = k + (c - vortCell);
                            rA = (c - vortCell) * dx;   // z offset
                            rB = (b - vortCell) * dx;   // x offset
                            break;
                        default:  // orient == 3, axis = z, radial plane = (x, y)
                            iloc = i + (b - vortCell);
                            jloc = j + (c - vortCell);
                            kloc = k + (a - vortCell);
                            rA = (b - vortCell) * dx;   // x offset
                            rB = (c - vortCell) * dx;   // y offset
                            break;
                    }
                    double r = Math.sqrt(rA * rA + rB * rB) + smallx;
                    double r2 = r * r;
                    double ur = -0.5 * alpha * r;
                    double uth = (gamma / (pi2 * r)) * (1.0 - Math.exp(aOver4nu * r2));
                    double costh = rA / r;
                    double sinth = rB / r;

                    int idx = ((iloc - 1) * n2 + (jloc - 1)) * n3 + (kloc - 1);
                    switch (orient) {
                        case 1:
                            u[idx] += uz;
                            v[idx] += costh * ur - sinth * uth;
                            w[idx] += sinth * ur + costh * uth;
                            break;
                        case 2:
                            v[idx] += uz;
                            w[idx] += costh * ur - sinth * uth;
                            u[idx] += sinth * ur + costh * uth;
                            break;
                        default:
                            w[idx] += uz;
                            u[idx] += costh * ur - sinth * uth;
                            v[idx] += sinth * ur + costh * uth;
                            break;
                    }
                }
            }
        }
    }

    /**
     * Zeros the six boundary planes of the three physical-space
     * velocity arrays so the subsequent FFT sees a periodic field —
     * a vortex tile can otherwise straddle a face and leave a
     * discontinuity there.
     */
    private static void zeroPlanes(double[] u, double[] v, double[] w, int n1, int n2, int n3) {
        // i = 0 and i = n1-1 planes
        for (int j = 0; j < n2; j++) {
            for (int k = 0; k < n3; k++) {
                int a = (0 * n2 + j) * n3 + k;
                int b = ((n1 - 1) * n2 + j) * n3 + k;
                u[a] = 0; u[b] = 0;
                v[a] = 0; v[b] = 0;
                w[a] = 0; w[b] = 0;
            }
        }
        // j = 0 and j = n2-1 planes
        for (int i = 0; i < n1; i++) {
            for (int k = 0; k < n3; k++) {
                int a = (i * n2 + 0) * n3 + k;
                int b = (i * n2 + (n2 - 1)) * n3 + k;
                u[a] = 0; u[b] = 0;
                v[a] = 0; v[b] = 0;
                w[a] = 0; w[b] = 0;
            }
        }
        // k = 0 and k = n3-1 planes
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                int a = (i * n2 + j) * n3 + 0;
                int b = (i * n2 + j) * n3 + (n3 - 1);
                u[a] = 0; u[b] = 0;
                v[a] = 0; v[b] = 0;
                w[a] = 0; w[b] = 0;
            }
        }
    }

    /**
     * Renormalizes the just-initialized Fourier spectrum so that the
     * physical-space kinetic energy per grid cell equals
     * {@code targetEnergy}.
     *
     * <p>Under the numpy-style FFT convention (forward unscaled, inverse
     * ×1/N), Parseval gives {@code Σ u²/N = Σ|v̂|²/N²}. So a Σ|v̂|² of
     * O(0.1) produces an energy of O(1/N²)
     * This method measures the actual initial energy via one round-trip through
     * {@code irfft3} and scales the whole spectrum by
     * {@code √(targetEnergy / measuredEnergy)} so the caller doesn't
     * have to reason about our FFT convention.
     */
    public static void rescaleInitialEnergy(NavierStokes3D sim, double targetEnergy) {
        // Measure the per-component physical energy and rescale each
        // component INDEPENDENTLY to hit targetEnergy/3. This does two
        // things at once:
        //   (1) drives the total physical energy to targetEnergy (as
        //       the previous global rescale did), and
        //   (2) enforces isotropy at t=0 by construction — a1 = a2 =
        //       a3 = 1/3 to machine precision — removing the Monte
        //       Carlo variance a single random realization would
        //       otherwise leave.
        // The divergence-free projection is amplitude-linear, so
        // scaling each velocity component's Fourier coefficients by
        // the same factor preserves solenoidality of that component's
        // contribution and does not mix components.
        NDArray uReal = Fft.irfft3(sim.getVeloXRe(), sim.getVeloXIm());
        NDArray vReal = Fft.irfft3(sim.getVeloYRe(), sim.getVeloYIm());
        NDArray wReal = Fft.irfft3(sim.getVeloZRe(), sim.getVeloZIm());
        double[] u = uReal.toDoubleArray();
        double[] v = vReal.toDoubleArray();
        double[] w = wReal.toDoubleArray();
        double sumU2 = 0.0, sumV2 = 0.0, sumW2 = 0.0;
        for (int i = 0; i < u.length; i++) {
            sumU2 += u[i] * u[i];
            sumV2 += v[i] * v[i];
            sumW2 += w[i] * w[i];
        }
        int n = u.length;
        double eU = sumU2 / n, eV = sumV2 / n, eW = sumW2 / n;
        double measuredEnergy = eU + eV + eW;
        double targetPerComp = targetEnergy / 3.0;
        if (eU < 1e-300 || eV < 1e-300 || eW < 1e-300) {
            logger.warn("At least one component's measured energy is essentially zero; skipping rescale.");
            return;
        }
        double fU = Math.sqrt(targetPerComp / eU);
        double fV = Math.sqrt(targetPerComp / eV);
        double fW = Math.sqrt(targetPerComp / eW);
        logger.info("Rescaling initial flow to E={} and enforcing isotropy: " +
                    "measured Eu/Ev/Ew = {}/{}/{}, factors = {}/{}/{}",
            String.format("%.4e", targetEnergy),
            String.format("%.4e", eU), String.format("%.4e", eV), String.format("%.4e", eW),
            String.format("%.4f", fU), String.format("%.4f", fV), String.format("%.4f", fW));

        scaleInPlace(sim.getVeloXRe(), fU);
        scaleInPlace(sim.getVeloXIm(), fU);
        scaleInPlace(sim.getVeloYRe(), fV);
        scaleInPlace(sim.getVeloYIm(), fV);
        scaleInPlace(sim.getVeloZRe(), fW);
        scaleInPlace(sim.getVeloZIm(), fW);
    }

    private static void scaleInPlace(NDArray a, double factor) {
        double[] data = a.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] *= factor;
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
