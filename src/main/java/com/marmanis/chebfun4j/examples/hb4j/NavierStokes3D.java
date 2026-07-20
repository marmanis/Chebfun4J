package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.DType;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Properties;

/**
 * 3-D Periodic Incompressible Navier-Stokes Spectral Simulation (HB4J).
 * 
 * <p>This class solves the Navier-Stokes equations in rotational form under the 
 * incompressibility constraint on a cubic periodic domain of length 2pi.
 * It is a high-performance Java port of the Fortran spectral code "HyPeRBoX".
 * 
 * <p>The time advancement is performed using a stiffly stable scheme of order 1, 2, or 3, 
 * with implicit treatment of the linear terms (diffusion) and explicit projection for the 
 * divergence-free constraint in Fourier space.
 */
public class NavierStokes3D {

    private static final Logger logger = LoggerFactory.getLogger(NavierStokes3D.class);

    // Device on which the velocity/vorticity NDArrays live. When set to a
    // GPU device with cuFFT available, Fft.rfft3/irfft3 auto-dispatch to
    // cuFFT (see Fft.java's cuFFT bridge). Defaults to host — call
    // {@link #setDevice(Device)} before {@link #initialize} to enable GPU.
    private Device device = Device.host();

    // Grid resolution parameters
    private final int n1, n2, n3;
    private final int half; // Size of the conjugate-symmetric Fourier half-dimension (n3 / 2 + 1)
    private final double fx, fy, fz;
    private final double dx, dx2;

    // Simulation configuration parameters (loaded from properties or configured dynamically)
    private boolean newflo = true;
    private boolean scaleFlow = false;
    private boolean traveling = false;
    private boolean firstOrder = false;
    private boolean storeFile = true;
    private boolean plank = false;
    private int initialFlow = 1; // 1 = random flow (ranflow), 2 = vortex cell flow (vortexFlow)
    private double filterRhalf = 0.5;
    private double cutoff = 0.85; // Spectral cutoff radius for de-aliasing filter
    private double facp = 2.5;
    private double scaleEnergy = 1.0;
    private double rnu = 0.0001; // Kinematic viscosity
    private double uForce = 0.0;
    private double vAlpha = 0.05;
    private double vBeta = 0.5;
    private double vortexStrength = 0.015;
    private int vCell = 6;
    private int vStep = 6;
    private double vRatio = 0.2;
    private int nhalt = 1000; // Number of timesteps to run
    private int nspec = 100;
    private int nshort = 10; // Number of steps between statistics logging
    private double ttime = 0.0; // Current simulation time
    private int norder = 2; // Stiffly stable marching scheme order (1, 2, or 3)
    private double dt = 0.0001; // Timestep size
    private double c1 = 1.0, c2 = -1.0, c3 = -1.0, c4 = 1.0; // Random flow spectrum constants

    // Velocity fields in Fourier space (represented as real/imaginary NDArray pairs of shape (n1, n2, half))
    private NDArray veloXRe, veloXIm;
    private NDArray veloYRe, veloYIm;
    private NDArray veloZRe, veloZIm;

    // Historical velocity fields for multi-step stiffly stable marching
    private NDArray pVeloXRe, pVeloXIm, pVeloYRe, pVeloYIm, pVeloZRe, pVeloZIm;
    private NDArray ppVeloXRe, ppVeloXIm, ppVeloYRe, ppVeloYIm, ppVeloZRe, ppVeloZIm;

    // Historical Lamb vector fields
    private NDArray pLambXRe, pLambXIm, pLambYRe, pLambYIm, pLambZRe, pLambZIm;
    private NDArray ppLambXRe, ppLambXIm, ppLambYRe, ppLambYIm, ppLambZRe, ppLambZIm;

    // Precomputed wavenumber variables
    private double[] kx, ky, kz;
    private double[] kx2, ky2, kz2;
    private double[] filter1; // De-aliasing filter matrix

    /**
     * Constructs the 3-D simulation grid.
     *
     * @param n1 grid resolution along the x-axis
     * @param n2 grid resolution along the y-axis
     * @param n3 grid resolution along the z-axis
     */
    public NavierStokes3D(int n1, int n2, int n3) {
        this.n1 = n1;
        this.n2 = n2;
        this.n3 = n3;
        this.half = n3 / 2 + 1;

        double pi2 = 2 * Math.PI;
        double hx = pi2 * ((double) n1 / n2);
        double hy = pi2;
        double hz = pi2;

        this.fx = pi2 / hx;
        this.fy = pi2 / hy;
        this.fz = pi2 / hz;

        this.dx = pi2 / n2;
        this.dx2 = 2.0 * dx;

        initWavenumbers();
    }

    /**
     * Route the velocity, vorticity, and Lamb-vector spectra through the
     * given device. When {@code d} is a GPU with a working cuFFT install,
     * every {@code rfft3}/{@code irfft3} call auto-dispatches to cuFFT.
     * Must be called before {@link #initialize} or {@link #ranflow}.
     */
    public void setDevice(Device d) {
        this.device = (d != null) ? d : Device.host();
    }

    public Device getDevice() {
        return device;
    }

    private NDArray wrap(double[] data, Shape shape) {
        return new ConcreteNDArray(data, shape, device);
    }

    private void initWavenumbers() {
        kx = new double[n1];
        kx2 = new double[n1];
        for (int i = 0; i < n1; i++) {
            int k = (i <= n1 / 2) ? i : i - n1;
            kx[i] = k * fx;
            kx2[i] = kx[i] * kx[i];
        }

        ky = new double[n2];
        ky2 = new double[n2];
        for (int j = 0; j < n2; j++) {
            int k = (j <= n2 / 2) ? j : j - n2;
            ky[j] = k * fy;
            ky2[j] = ky[j] * ky[j];
        }

        kz = new double[half];
        kz2 = new double[half];
        for (int k = 0; k < half; k++) {
            kz[k] = k * fz;
            kz2[k] = kz[k] * kz[k];
        }
    }

    /**
     * Loads simulation parameters from a standard Java properties file.
     *
     * @param path path to the properties file
     */
    public void loadProperties(String path) {
        File file = new File(path);
        if (!file.exists()) {
            logger.warn("Properties file not found: {}. Using default parameters.", path);
            return;
        }
        Properties props = new Properties();
        try (InputStream in = new FileInputStream(file)) {
            props.load(in);
            newflo = Boolean.parseBoolean(props.getProperty("simulation.newflow", String.valueOf(newflo)));
            scaleFlow = Boolean.parseBoolean(props.getProperty("simulation.scaleflow", String.valueOf(scaleFlow)));
            traveling = Boolean.parseBoolean(props.getProperty("simulation.traveling", String.valueOf(traveling)));
            firstOrder = Boolean.parseBoolean(props.getProperty("simulation.firstOrder", String.valueOf(firstOrder)));
            storeFile = Boolean.parseBoolean(props.getProperty("simulation.storeFile", String.valueOf(storeFile)));
            plank = Boolean.parseBoolean(props.getProperty("simulation.plank", String.valueOf(plank)));
            initialFlow = Integer.parseInt(props.getProperty("simulation.initialFlow", String.valueOf(initialFlow)));
            filterRhalf = Double.parseDouble(props.getProperty("simulation.filterRhalf", String.valueOf(filterRhalf)));
            cutoff = Double.parseDouble(props.getProperty("simulation.cutoff", String.valueOf(cutoff)));
            facp = Double.parseDouble(props.getProperty("simulation.facp", String.valueOf(facp)));
            scaleEnergy = Double.parseDouble(props.getProperty("simulation.scaleEnergy", String.valueOf(scaleEnergy)));
            rnu = Double.parseDouble(props.getProperty("simulation.rnu", String.valueOf(rnu)));
            uForce = Double.parseDouble(props.getProperty("simulation.uForce", String.valueOf(uForce)));
            vAlpha = Double.parseDouble(props.getProperty("simulation.vAlpha", String.valueOf(vAlpha)));
            vBeta = Double.parseDouble(props.getProperty("simulation.vBeta", String.valueOf(vBeta)));
            vortexStrength = Double.parseDouble(props.getProperty("simulation.vortexStrength", String.valueOf(vortexStrength)));
            vCell = Integer.parseInt(props.getProperty("simulation.vCell", String.valueOf(vCell)));
            vStep = Integer.parseInt(props.getProperty("simulation.vStep", String.valueOf(vStep)));
            vRatio = Double.parseDouble(props.getProperty("simulation.vRatio", String.valueOf(vRatio)));
            nhalt = Integer.parseInt(props.getProperty("simulation.nhalt", String.valueOf(nhalt)));
            nspec = Integer.parseInt(props.getProperty("simulation.nspec", String.valueOf(nspec)));
            nshort = Integer.parseInt(props.getProperty("simulation.nshort", String.valueOf(nshort)));
            ttime = Double.parseDouble(props.getProperty("simulation.ttime", String.valueOf(ttime)));
            norder = Integer.parseInt(props.getProperty("simulation.norder", String.valueOf(norder)));
            dt = Double.parseDouble(props.getProperty("simulation.dt", String.valueOf(dt)));
            c1 = Double.parseDouble(props.getProperty("simulation.c1", String.valueOf(c1)));
            c2 = Double.parseDouble(props.getProperty("simulation.c2", String.valueOf(c2)));
            c3 = Double.parseDouble(props.getProperty("simulation.c3", String.valueOf(c3)));
            c4 = Double.parseDouble(props.getProperty("simulation.c4", String.valueOf(c4)));
            logger.info("Properties successfully loaded from: {}", path);
        } catch (IOException e) {
            logger.error("Error reading properties file: {}", e.getMessage(), e);
        }
    }

    /**
     * Saves the current simulation velocity fields to a binary checkpoint file.
     *
     * @param path checkpoint file path
     */
    public void saveCheckpoint(String path) {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            out.writeInt(n1);
            out.writeInt(n2);
            out.writeInt(n3);
            out.writeDouble(ttime);

            double[] vxRe = veloXRe.toDoubleArray();
            double[] vxIm = veloXIm.toDoubleArray();
            double[] vyRe = veloYRe.toDoubleArray();
            double[] vyIm = veloYIm.toDoubleArray();
            double[] vzRe = veloZRe.toDoubleArray();
            double[] vzIm = veloZIm.toDoubleArray();

            for (double v : vxRe) out.writeDouble(v);
            for (double v : vxIm) out.writeDouble(v);
            for (double v : vyRe) out.writeDouble(v);
            for (double v : vyIm) out.writeDouble(v);
            for (double v : vzRe) out.writeDouble(v);
            for (double v : vzIm) out.writeDouble(v);

            logger.info("Checkpoint saved successfully to: {}", path);
        } catch (IOException e) {
            logger.error("Error writing checkpoint: {}", e.getMessage(), e);
        }
    }

    /**
     * Restores the velocity fields and simulation time from a binary checkpoint file.
     *
     * @param path checkpoint file path
     */
    public void loadCheckpoint(String path) {
        File file = new File(path);
        if (!file.exists()) {
            logger.warn("Checkpoint file not found: {}. Initializing with random flow fallback.", path);
            FlowFieldInitializer.initializeRandomFlow(this);
            return;
        }
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
            int fileN1 = in.readInt();
            int fileN2 = in.readInt();
            int fileN3 = in.readInt();
            if (fileN1 != n1 || fileN2 != n2 || fileN3 != n3) {
                throw new IllegalArgumentException(String.format("Checkpoint grid dimensions mismatch: (%d,%d,%d) in file vs (%d,%d,%d) configured", fileN1, fileN2, fileN3, n1, n2, n3));
            }
            ttime = in.readDouble();

            Shape shape = new Shape(n1, n2, half);
            int size = n1 * n2 * half;

            double[] vxRe = new double[size];
            double[] vxIm = new double[size];
            double[] vyRe = new double[size];
            double[] vyIm = new double[size];
            double[] vzRe = new double[size];
            double[] vzIm = new double[size];

            for (int i = 0; i < size; i++) vxRe[i] = in.readDouble();
            for (int i = 0; i < size; i++) vxIm[i] = in.readDouble();
            for (int i = 0; i < size; i++) vyRe[i] = in.readDouble();
            for (int i = 0; i < size; i++) vyIm[i] = in.readDouble();
            for (int i = 0; i < size; i++) vzRe[i] = in.readDouble();
            for (int i = 0; i < size; i++) vzIm[i] = in.readDouble();

            veloXRe = wrap(vxRe, shape);
            veloXIm = wrap(vxIm, shape);
            veloYRe = wrap(vyRe, shape);
            veloYIm = wrap(vyIm, shape);
            veloZRe = wrap(vzRe, shape);
            veloZIm = wrap(vzIm, shape);

            logger.info("Checkpoint loaded successfully from: {} at ttime = {}", path, ttime);
        } catch (IOException e) {
            logger.error("Error reading checkpoint: {}", e.getMessage(), e);
            logger.info("Initializing with random flow as fallback.");
            FlowFieldInitializer.initializeRandomFlow(this);
        }
    }

    /**
     * Initializes the simulation arrays and loads either the configured initial flow
     * or a checkpoint for a restart run.
     */
    public void initialize() {
        initFilter();

        Shape shape = new Shape(n1, n2, half);
        double[] zeros = new double[n1 * n2 * half];

        veloXRe = wrap(zeros.clone(), shape);
        veloXIm = wrap(zeros.clone(), shape);
        veloYRe = wrap(zeros.clone(), shape);
        veloYIm = wrap(zeros.clone(), shape);
        veloZRe = wrap(zeros.clone(), shape);
        veloZIm = wrap(zeros.clone(), shape);

        if (newflo) {
            if (initialFlow == 1) {
                FlowFieldInitializer.initializeRandomFlow(this);
            } else if (initialFlow == 2) {
                FlowFieldInitializer.initializeVortexFlow(this);
            }
        } else {
            logger.info("Attempting restart from checkpoint...");
            loadCheckpoint("checkpoint.dat");
        }
    }

    private void initFilter() {
        filter1 = new double[n1 * n2 * half];
        Arrays.fill(filter1, 1.0);
        if (cutoff < 1.0 && cutoff > 0.0) {
            double fkmax = kx[n1 / 2];
            double fkstar = cutoff * fkmax;
            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    for (int k = 0; k < half; k++) {
                        double fkst = ky2[j] + kz2[k];
                        double fkrt = Math.sqrt(kx2[i] + fkst);
                        int idx = (i * n2 + j) * half + k;
                        if (fkrt > fkstar) {
                            filter1[idx] = 0.5 * (Math.cos(Math.PI * (fkrt - fkstar) / (fkmax - fkstar)) + 1.0);
                        }
                        if (fkrt > fkmax) {
                            filter1[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /**
     * Executes the main simulation loop for nhalt timesteps.
     */
    public void run() {
        int J_order = 1;
        double alpha02 = 2.0, alpha12 = -0.5;
        double alpha03 = 3.0, alpha13 = -1.5, alpha23 = 1.0 / 3.0;
        double beta02 = 2.0;
        double beta03 = 3.0, beta13 = -3.0;
        double gamma0 = 1.0;

        Shape shape = new Shape(n1, n2, half);

        pVeloXRe = veloXRe; pVeloXIm = veloXIm;
        pVeloYRe = veloYRe; pVeloYIm = veloYIm;
        pVeloZRe = veloZRe; pVeloZIm = veloZIm;

        ppVeloXRe = veloXRe; ppVeloXIm = veloXIm;
        ppVeloYRe = veloYRe; ppVeloYIm = veloYIm;
        ppVeloZRe = veloZRe; ppVeloZIm = veloZIm;

        pLambXRe = veloXRe; pLambXIm = veloXIm;
        pLambYRe = veloYRe; pLambYIm = veloYIm;
        pLambZRe = veloZRe; pLambZIm = veloZIm;

        ppLambXRe = veloXRe; ppLambXIm = veloXIm;
        ppLambYRe = veloYRe; ppLambYIm = veloYIm;
        ppLambZRe = veloZRe; ppLambZIm = veloZIm;

        logger.info("Starting simulation loop for {} steps...", nhalt);
        logger.info(String.format("%10s %10s %10s %10s", "ttime", "a1", "a2", "a3"));

        for (int istep = 0; istep <= nhalt; istep++) {
            // 1. Compute Vorticity in Fourier space: w = i k x v
            double[] vxRe = veloXRe.toDoubleArray();
            double[] vxIm = veloXIm.toDoubleArray();
            double[] vyRe = veloYRe.toDoubleArray();
            double[] vyIm = veloYIm.toDoubleArray();
            double[] vzRe = veloZRe.toDoubleArray();
            double[] vzIm = veloZIm.toDoubleArray();

            double[] wxRe = new double[n1 * n2 * half];
            double[] wxIm = new double[n1 * n2 * half];
            double[] wyRe = new double[n1 * n2 * half];
            double[] wyIm = new double[n1 * n2 * half];
            double[] wzRe = new double[n1 * n2 * half];
            double[] wzIm = new double[n1 * n2 * half];

            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    for (int k = 0; k < half; k++) {
                        int idx = (i * n2 + j) * half + k;
                        wxRe[idx] = -ky[j] * vzIm[idx] + kz[k] * vyIm[idx];
                        wxIm[idx] =  ky[j] * vzRe[idx] - kz[k] * vyRe[idx];

                        wyRe[idx] = -kz[k] * vxIm[idx] + kx[i] * vzIm[idx];
                        wyIm[idx] =  kz[k] * vxRe[idx] - kx[i] * vzRe[idx];

                        wzRe[idx] = -kx[i] * vyIm[idx] + ky[j] * vxIm[idx];
                        wzIm[idx] =  kx[i] * vyRe[idx] - ky[j] * vxRe[idx];
                    }
                }
            }

            // 2. Inverse FFT velocity and vorticity back to physical real space
            NDArray uReal = Fft.irfft3(veloXRe, veloXIm);
            NDArray vReal = Fft.irfft3(veloYRe, veloYIm);
            NDArray wReal = Fft.irfft3(veloZRe, veloZIm);

            NDArray wxReal = Fft.irfft3(wrap(wxRe, shape), wrap(wxIm, shape));
            NDArray wyReal = Fft.irfft3(wrap(wyRe, shape), wrap(wyIm, shape));
            NDArray wzReal = Fft.irfft3(wrap(wzRe, shape), wrap(wzIm, shape));

            // 3. Compute Lamb vector in physical space: L = v x w
            double[] uData = uReal.toDoubleArray();
            double[] vData = vReal.toDoubleArray();
            double[] wData = wReal.toDoubleArray();
            double[] wxData = wxReal.toDoubleArray();
            double[] wyData = wyReal.toDoubleArray();
            double[] wzData = wzReal.toDoubleArray();

            int realSize = n1 * n2 * n3;
            double[] LxData = new double[realSize];
            double[] LyData = new double[realSize];
            double[] LzData = new double[realSize];

            for (int idx = 0; idx < realSize; idx++) {
                LxData[idx] = vData[idx] * wzData[idx] - wData[idx] * wyData[idx];
                LyData[idx] = wData[idx] * wxData[idx] - uData[idx] * wzData[idx];
                LzData[idx] = uData[idx] * wyData[idx] - vData[idx] * wxData[idx];
            }

            // Logging short statistics
            if (istep % nshort == 0) {
                FlowStatistics.logShortStats(ttime, n1, n2, n3, uData, vData, wData);
            }

            // 5. Forward FFT Lamb vector to Fourier space
            NDArray[] LxSpec = Fft.rfft3(wrap(LxData, new Shape(n1, n2, n3)));
            NDArray[] LySpec = Fft.rfft3(wrap(LyData, new Shape(n1, n2, n3)));
            NDArray[] LzSpec = Fft.rfft3(wrap(LzData, new Shape(n1, n2, n3)));

            NDArray LxRe = LxSpec[0], LxIm = LxSpec[1];
            NDArray LyRe = LySpec[0], LyIm = LySpec[1];
            NDArray LzRe = LzSpec[0], LzIm = LzSpec[1];

            // Extract spectra
            if (istep % nspec == 0) {
                FlowStatistics.computeAndWriteSpectra("Spectra.dat", rnu, ttime, n1, n2, n3, half,
                                                      kx2, ky2, kz2,
                                                      veloXRe, veloXIm, veloYRe, veloYIm, veloZRe, veloZIm,
                                                      LxRe, LxIm, LyRe, LyIm, LzRe, LzIm);
            }

            // 6. March velocity field
            NDArray nextVeloXRe, nextVeloXIm;
            NDArray nextVeloYRe, nextVeloYIm;
            NDArray nextVeloZRe, nextVeloZIm;

            switch (J_order) {
                case 1 -> {
                    nextVeloXRe = veloXRe.add(LxRe.mul(scalar(dt)));
                    nextVeloXIm = veloXIm.add(LxIm.mul(scalar(dt)));

                    nextVeloYRe = veloYRe.add(LyRe.mul(scalar(dt)));
                    nextVeloYIm = veloYIm.add(LyIm.mul(scalar(dt)));

                    nextVeloZRe = veloZRe.add(LzRe.mul(scalar(dt)));
                    nextVeloZIm = veloZIm.add(LzIm.mul(scalar(dt)));

                    pVeloXRe = veloXRe; pVeloXIm = veloXIm;
                    pVeloYRe = veloYRe; pVeloYIm = veloYIm;
                    pVeloZRe = veloZRe; pVeloZIm = veloZIm;

                    pLambXRe = LxRe; pLambXIm = LxIm;
                    pLambYRe = LyRe; pLambYIm = LyIm;
                    pLambZRe = LzRe; pLambZIm = LzIm;

                    gamma0 = 1.0;
                    if (!firstOrder) J_order = 2;
                }
                case 2 -> {
                    if (norder == 3) {
                        ppVeloXRe = pVeloXRe; ppVeloXIm = pVeloXIm;
                        ppVeloYRe = pVeloYRe; ppVeloYIm = pVeloYIm;
                        ppVeloZRe = pVeloZRe; ppVeloZIm = pVeloZIm;

                        ppLambXRe = pLambXRe; ppLambXIm = pLambXIm;
                        ppLambYRe = pLambYRe; ppLambYIm = pLambYIm;
                        ppLambZRe = pLambZRe; ppLambZIm = pLambZIm;
                    }

                    nextVeloXRe = veloXRe.mul(scalar(alpha02)).add(pVeloXRe.mul(scalar(alpha12))).add(LxRe.mul(scalar(beta02 * dt)).sub(pLambXRe.mul(scalar(dt))));
                    nextVeloXIm = veloXIm.mul(scalar(alpha02)).add(pVeloXIm.mul(scalar(alpha12))).add(LxIm.mul(scalar(beta02 * dt)).sub(pLambXIm.mul(scalar(dt))));

                    nextVeloYRe = veloYRe.mul(scalar(alpha02)).add(pVeloYRe.mul(scalar(alpha12))).add(LyRe.mul(scalar(beta02 * dt)).sub(pLambYRe.mul(scalar(dt))));
                    nextVeloYIm = veloYIm.mul(scalar(alpha02)).add(pVeloYIm.mul(scalar(alpha12))).add(LyIm.mul(scalar(beta02 * dt)).sub(pLambYIm.mul(scalar(dt))));

                    nextVeloZRe = veloZRe.mul(scalar(alpha02)).add(pVeloZRe.mul(scalar(alpha12))).add(LzRe.mul(scalar(beta02 * dt)).sub(pLambZRe.mul(scalar(dt))));
                    nextVeloZIm = veloZIm.mul(scalar(alpha02)).add(pVeloZIm.mul(scalar(alpha12))).add(LzIm.mul(scalar(beta02 * dt)).sub(pLambZIm.mul(scalar(dt))));

                    pVeloXRe = veloXRe; pVeloXIm = veloXIm;
                    pVeloYRe = veloYRe; pVeloYIm = veloYIm;
                    pVeloZRe = veloZRe; pVeloZIm = veloZIm;

                    pLambXRe = LxRe; pLambXIm = LxIm;
                    pLambYRe = LyRe; pLambYIm = LyIm;
                    pLambZRe = LzRe; pLambZIm = LzIm;

                    gamma0 = 1.5;
                    J_order = norder;
                }
                case 3 -> {
                    nextVeloXRe = veloXRe.mul(scalar(alpha03)).add(pVeloXRe.mul(scalar(alpha13))).add(ppVeloXRe.mul(scalar(alpha23)))
                            .add(LxRe.mul(scalar(beta03 * dt)).add(pLambXRe.mul(scalar(beta13 * dt))).add(ppLambXRe.mul(scalar(dt))));
                    nextVeloXIm = veloXIm.mul(scalar(alpha03)).add(pVeloXIm.mul(scalar(alpha13))).add(ppVeloXIm.mul(scalar(alpha23)))
                            .add(LxIm.mul(scalar(beta03 * dt)).add(pLambXIm.mul(scalar(beta13 * dt))).add(ppLambXIm.mul(scalar(dt))));

                    nextVeloYRe = veloYRe.mul(scalar(alpha03)).add(pVeloYRe.mul(scalar(alpha13))).add(ppVeloYRe.mul(scalar(alpha23)))
                            .add(LyRe.mul(scalar(beta03 * dt)).add(pLambYRe.mul(scalar(beta13 * dt))).add(ppLambYRe.mul(scalar(dt))));
                    nextVeloYIm = veloYIm.mul(scalar(alpha03)).add(pVeloYIm.mul(scalar(alpha13))).add(ppVeloYIm.mul(scalar(alpha23)))
                            .add(LyIm.mul(scalar(beta03 * dt)).add(pLambYIm.mul(scalar(beta13 * dt))).add(ppLambYIm.mul(scalar(dt))));

                    nextVeloZRe = veloZRe.mul(scalar(alpha03)).add(pVeloZRe.mul(scalar(alpha13))).add(ppVeloZRe.mul(scalar(alpha23)))
                            .add(LzRe.mul(scalar(beta03 * dt)).add(pLambZRe.mul(scalar(beta13 * dt))).add(ppLambZRe.mul(scalar(dt))));
                    nextVeloZIm = veloZIm.mul(scalar(alpha03)).add(pVeloZIm.mul(scalar(alpha13))).add(ppVeloZIm.mul(scalar(alpha23)))
                            .add(LzIm.mul(scalar(beta03 * dt)).add(pLambZIm.mul(scalar(beta13 * dt))).add(ppLambZIm.mul(scalar(dt))));

                    ppVeloXRe = pVeloXRe; ppVeloXIm = pVeloXIm;
                    ppVeloYRe = pVeloYRe; ppVeloYIm = pVeloYIm;
                    ppVeloZRe = pVeloZRe; ppVeloZIm = pVeloZIm;

                    pVeloXRe = veloXRe; pVeloXIm = veloXIm;
                    pVeloYRe = veloYRe; pVeloYIm = veloYIm;
                    pVeloZRe = veloZRe; pVeloZIm = veloZIm;

                    ppLambXRe = pLambXRe; ppLambXIm = pLambXIm;
                    ppLambYRe = pLambYRe; ppLambYIm = pLambYIm;
                    ppLambZRe = ppLambZRe; ppLambZIm = ppLambZIm;

                    pLambXRe = LxRe; pLambXIm = LxIm;
                    pLambYRe = LyRe; pLambYIm = LyIm;
                    pLambZRe = LzRe; pLambZIm = LzIm;

                    gamma0 = 11.0 / 6.0;
                }
                default -> throw new IllegalArgumentException("Invalid J_order: " + J_order);
            }

            // 7. Linear terms (implicit diffusion and pressure projection to maintain zero divergence)
            double[] nvxRe = nextVeloXRe.toDoubleArray();
            double[] nvxIm = nextVeloXIm.toDoubleArray();
            double[] nvyRe = nextVeloYRe.toDoubleArray();
            double[] nvyIm = nextVeloYIm.toDoubleArray();
            double[] nvzRe = nextVeloZRe.toDoubleArray();
            double[] nvzIm = nextVeloZIm.toDoubleArray();

            double rnuf = rnu * dt;
            double smallx = 1.0e-14;

            for (int i = 0; i < n1; i++) {
                for (int j = 0; j < n2; j++) {
                    for (int k = 0; k < half; k++) {
                        double temp1 = kx2[i] + ky2[j] + kz2[k] + smallx;
                        int idx = (i * n2 + j) * half + k;

                        double temp2Re = (kx[i] * nvxRe[idx] + ky[j] * nvyRe[idx] + kz[k] * nvzRe[idx]) / temp1;
                        double temp2Im = (kx[i] * nvxIm[idx] + ky[j] * nvyIm[idx] + kz[k] * nvzIm[idx]) / temp1;

                        double temp3 = 1.0 / (gamma0 + rnuf * temp1);

                        nvxRe[idx] = temp3 * (nvxRe[idx] - kx[i] * temp2Re);
                        nvxIm[idx] = temp3 * (nvxIm[idx] - kx[i] * temp2Im);

                        nvyRe[idx] = temp3 * (nvyRe[idx] - ky[j] * temp2Re);
                        nvyIm[idx] = temp3 * (nvyIm[idx] - ky[j] * temp2Im);

                        nvzRe[idx] = temp3 * (nvzRe[idx] - kz[k] * temp2Re);
                        nvzIm[idx] = temp3 * (nvzIm[idx] - kz[k] * temp2Im);
                    }
                }
            }

            cleanSymmetries(nvxRe, nvxIm);
            cleanSymmetries(nvyRe, nvyIm);
            cleanSymmetries(nvzRe, nvzIm);

            nvxRe[0] = 0.0; nvxIm[0] = 0.0;
            nvyRe[0] = 0.0; nvyIm[0] = 0.0;
            nvzRe[0] = 0.0; nvzIm[0] = 0.0;

            if (cutoff < 1.0) {
                for (int idx = 0; idx < filter1.length; idx++) {
                    nvxRe[idx] *= filter1[idx];
                    nvxIm[idx] *= filter1[idx];
                    nvyRe[idx] *= filter1[idx];
                    nvyIm[idx] *= filter1[idx];
                    nvzRe[idx] *= filter1[idx];
                    nvzIm[idx] *= filter1[idx];
                }
            }

            veloXRe = wrap(nvxRe, shape);
            veloXIm = wrap(nvxIm, shape);
            veloYRe = wrap(nvyRe, shape);
            veloYIm = wrap(nvyIm, shape);
            veloZRe = wrap(nvzRe, shape);
            veloZIm = wrap(nvzIm, shape);

            ttime += dt;
        }

        logger.info("Simulation loop finished.");
        if (storeFile) {
            saveCheckpoint("checkpoint.dat");
        }
        FlowStatistics.showPlots();
    }



    public int getN1() { return n1; }
    public int getN2() { return n2; }
    public int getHalf() { return half; }
    public double[] getKx() { return kx; }
    public double[] getKy() { return ky; }
    public double[] getKz() { return kz; }
    public double[] getKx2() { return kx2; }
    public double[] getKy2() { return ky2; }
    public double[] getKz2() { return kz2; }
    public double getC1() { return c1; }
    public double getC2() { return c2; }
    public double getC3() { return c3; }
    public double getC4() { return c4; }
    public double getFacp() { return facp; }
    public boolean isPlank() { return plank; }

    public void setVelocityFields(NDArray vxRe, NDArray vxIm, NDArray vyRe, NDArray vyIm, NDArray vzRe, NDArray vzIm) {
        this.veloXRe = vxRe;
        this.veloXIm = vxIm;
        this.veloYRe = vyRe;
        this.veloYIm = vyIm;
        this.veloZRe = vzRe;
        this.veloZIm = vzIm;
    }

    private void cleanSymmetries(double[] re, double[] im) {
        int nyq = half - 1;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                int idx = (i * n2 + j) * half + nyq;
                re[idx] = 0.0;
                im[idx] = 0.0;
            }
        }
    }

    private NDArray scalar(double val) {
        return wrap(new double[]{val}, new Shape(1));
    }

    public static void main(String[] args) {
        logger.info("=== Running HB4J 3-D Incompressible Navier-Stokes Spectral Simulation ===");

        // Device selection: --gpu (or HB4J_DEVICE=gpu) routes the FFT
        // buffers through Device.defaultDevice(), which cuFFT then picks
        // up automatically. Without the flag we stay on the host so the
        // test suite and non-GPU runs behave as before.
        boolean useGpu = false;
        for (String a : args) if ("--gpu".equalsIgnoreCase(a)) useGpu = true;
        String envDev = System.getenv("HB4J_DEVICE");
        if (envDev != null && envDev.equalsIgnoreCase("gpu")) useGpu = true;

        NavierStokes3D sim = new NavierStokes3D(64, 64, 64);
        if (useGpu) {
            Device d = Device.defaultDevice();
            sim.setDevice(d);
            logger.info("GPU mode requested: dispatching FFTs on {}", d);
        } else {
            logger.info("Host mode (pass --gpu to route FFTs through cuFFT)");
        }
        sim.loadProperties("src/test/resources/simulation.properties");
        sim.initialize();
        sim.run();
    }
}
