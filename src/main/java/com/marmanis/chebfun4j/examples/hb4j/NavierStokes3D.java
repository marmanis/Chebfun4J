package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.api.Fft;
import com.marmanis.jax4j.core.ConcreteNDArray;
import com.marmanis.jax4j.core.Device;
import com.marmanis.jax4j.core.NDArray;
import com.marmanis.jax4j.core.Shape;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Arrays;
import java.util.Properties;

import static com.marmanis.chebfun4j.util.Setup.envIntOr;
import static com.marmanis.chebfun4j.util.Setup.envIntOrNull;
import static com.marmanis.chebfun4j.util.Setup.envLongOrNull;

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
 *
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
    // Default ON: without this, the physical energy is O(1/N²) — machine
    // epsilon at typical grid sizes — because ranflow generates
    // Fourier coefficients on an O(0.1) scale that is only meaningful
    // under a different FFT-normalization convention. See
    // FlowFieldInitializer.rescaleInitialEnergy.
    private boolean scaleFlow = true;
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
    private double[] vAlpha = {0.05};
    private double[] vBeta = {0.5};
    private double[] vortexStrength = {0.015};
    private double vortexTubeSeparation = 1.0;
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
    // Seed for the initial random-flow spectrum. Exposed as
    // HB4J_SEED so a run can be re-realized under a different draw
    // without a source change — useful for averaging isotropy
    // diagnostics over independent realizations.
    private long randomSeed = -27343L;

    // Path of the checkpoint file to restart FROM when newflo=false.
    // Populated from the simulation.oldflow.file property; can also be
    // set via a positional arg to main(). Required when newflo=false —
    // no default filename, since silently loading a stale
    // "checkpoint.dat" that happens to be in the cwd would be a nasty
    // way to get a run that doesn't reflect the current properties.
    private String oldflowFile = null;

    // Short (8-hex-char) SHA-256 of the loaded properties-file bytes.
    // Baked into checkpoint filenames so a saved state is trivially
    // paired with the config that produced it; two runs of the same
    // properties file share an ID even across timestamps, and two
    // runs with different tweaks can never accidentally collide.
    private String simulationID = "nosim";

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
            vAlpha = parseDoubleArray(props.getProperty("simulation.vAlpha"), vAlpha);
            vBeta = parseDoubleArray(props.getProperty("simulation.vBeta"), vBeta);
            vortexStrength = parseDoubleArray(props.getProperty("simulation.vortexStrength"), vortexStrength);
            vortexTubeSeparation = Double.parseDouble(props.getProperty("simulation.vortexTubeSeparation", String.valueOf(vortexTubeSeparation)));
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
            String ofp = props.getProperty("simulation.oldflow.file");
            if (ofp != null && !ofp.trim().isEmpty()) oldflowFile = ofp.trim();
            simulationID = shortHashOf(file);
            logger.info("Properties successfully loaded from: {}", path);
        } catch (IOException e) {
            logger.error("Error reading properties file: {}", e.getMessage(), e);
        }
    }

    /**
     * First 8 hex chars of the SHA-256 of the file bytes — a stable,
     * config-derived short identifier for the run. Falls back to
     * {@code "nosim"} on any error so a broken hash never blocks the run.
     */
    private static String shortHashOf(File file) {
        try {
            java.security.MessageDigest md = java.security.MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(java.nio.file.Files.readAllBytes(file.toPath()));
            StringBuilder sb = new StringBuilder(8);
            for (int i = 0; i < 4; i++) sb.append(String.format("%02x", hash[i]));
            return sb.toString();
        } catch (Exception e) {
            logger.warn("Could not hash {} for simulationID: {}", file, e.getMessage());
            return "nosim";
        }
    }

    /**
     * Sets an explicit path to the checkpoint file to restart FROM.
     * Overrides {@code simulation.oldflow.file}. Useful when calling
     * {@code main()} with a positional argument, or programmatically
     * from a driver script.
     */
    public void setOldflowFile(String path) { this.oldflowFile = path; }

    public String getSimulationID() { return simulationID; }

    private static double[] parseDoubleArray(String raw, double[] fallback) {
        if (raw == null || raw.isBlank()) return fallback;
        String[] parts = raw.split(",");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Double.parseDouble(parts[i].trim());
        return out;
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
        if (path.toLowerCase(java.util.Locale.ROOT).endsWith(".csv")) {
            loadCsvCheckpoint(file);
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
     * Loads the initial flow from a CSV file. Each row must have exactly
     * 9 whitespace- or comma-separated numbers:
     * <pre>
     *     i  j  k   u  v  w   wx  wy  wz
     * </pre>
     * where {@code (i, j, k)} are 1-based grid indices in [1, N], the
     * middle triple is the velocity at that grid point, and the last
     * triple is the vorticity (used for validation only). Blank lines
     * and lines whose first non-whitespace character is '#' are ignored.
     *
     * <p>The file must contain exactly N³ rows, one per grid cell, in
     * any order. The grid size N is inferred from the maximum index and
     * must match {@code n1 = n2 = n3}. After loading, the velocity is
     * transformed to Fourier space; the vorticity from columns 7–9 is
     * compared against the curl computed from the loaded velocity, and
     * max-abs / RMS differences are logged.
     */
    private void loadCsvCheckpoint(File file) {
        if (n1 != n2 || n2 != n3) {
            throw new IllegalStateException(String.format(
                "CSV loader requires a cubic grid (n1=n2=n3); got %dx%dx%d",
                n1, n2, n3));
        }
        int N = n1;
        int total = N * N * N;

        double[] u  = new double[total];
        double[] v  = new double[total];
        double[] w  = new double[total];
        double[] wx = new double[total];
        double[] wy = new double[total];
        double[] wz = new double[total];
        boolean[] filled = new boolean[total];

        int rowCount = 0;
        int detectedMaxIdx = 0;
        try (BufferedReader in = new BufferedReader(new FileReader(file))) {
            String line;
            int lineNo = 0;
            while ((line = in.readLine()) != null) {
                lineNo++;
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) continue;
                // Skip header lines whose first token isn't numeric.
                String[] tok = trimmed.split("[,\\s]+");
                if (tok.length < 9) {
                    throw new IOException(String.format(
                        "CSV line %d: expected 9 fields (i j k u v w wx wy wz), got %d",
                        lineNo, tok.length));
                }
                int i, j, k;
                try {
                    i = Integer.parseInt(tok[0]);
                    j = Integer.parseInt(tok[1]);
                    k = Integer.parseInt(tok[2]);
                } catch (NumberFormatException nfe) {
                    // Non-numeric first token — treat as a header row and skip.
                    if (rowCount == 0) continue;
                    throw new IOException(String.format(
                        "CSV line %d: expected integer indices (i j k), got '%s %s %s'",
                        lineNo, tok[0], tok[1], tok[2]));
                }
                if (i < 1 || j < 1 || k < 1) {
                    throw new IOException(String.format(
                        "CSV line %d: 1-based indices must be >= 1; got (%d, %d, %d)",
                        lineNo, i, j, k));
                }
                if (i > N || j > N || k > N) {
                    throw new IOException(String.format(
                        "CSV line %d: index (%d, %d, %d) out of range for N=%d",
                        lineNo, i, j, k, N));
                }
                detectedMaxIdx = Math.max(detectedMaxIdx, Math.max(i, Math.max(j, k)));

                int i0 = i - 1, j0 = j - 1, k0 = k - 1;
                int idx = (i0 * N + j0) * N + k0;
                if (filled[idx]) {
                    throw new IOException(String.format(
                        "CSV line %d: duplicate grid cell (i=%d, j=%d, k=%d)",
                        lineNo, i, j, k));
                }
                u[idx]  = Double.parseDouble(tok[3]);
                v[idx]  = Double.parseDouble(tok[4]);
                w[idx]  = Double.parseDouble(tok[5]);
                wx[idx] = Double.parseDouble(tok[6]);
                wy[idx] = Double.parseDouble(tok[7]);
                wz[idx] = Double.parseDouble(tok[8]);
                filled[idx] = true;
                rowCount++;
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to read CSV checkpoint " + file + ": " + e.getMessage(), e);
        }

        if (rowCount != total) {
            throw new IllegalArgumentException(String.format(
                "CSV checkpoint %s: expected %d rows (N=%d, cubic), got %d",
                file, total, N, rowCount));
        }
        if (detectedMaxIdx != N) {
            throw new IllegalArgumentException(String.format(
                "CSV checkpoint %s: detected max index %d but simulation grid N=%d",
                file, detectedMaxIdx, N));
        }

        Shape realShape = new Shape(N, N, N);
        NDArray[] uSpec = Fft.rfft3(new ConcreteNDArray(u, realShape));
        NDArray[] vSpec = Fft.rfft3(new ConcreteNDArray(v, realShape));
        NDArray[] wSpec = Fft.rfft3(new ConcreteNDArray(w, realShape));

        Shape spectralShape = new Shape(n1, n2, half);
        veloXRe = wrap(uSpec[0].toDoubleArray(), spectralShape);
        veloXIm = wrap(uSpec[1].toDoubleArray(), spectralShape);
        veloYRe = wrap(vSpec[0].toDoubleArray(), spectralShape);
        veloYIm = wrap(vSpec[1].toDoubleArray(), spectralShape);
        veloZRe = wrap(wSpec[0].toDoubleArray(), spectralShape);
        veloZIm = wrap(wSpec[1].toDoubleArray(), spectralShape);

        ttime = 0.0;
        logger.info("CSV checkpoint loaded: {} rows from {} (ttime reset to 0)", rowCount, file);

        compareVorticityAgainstCurl(wx, wy, wz);
    }

    /**
     * Computes vorticity ω = ∇×u in Fourier space from the currently
     * loaded velocity fields, transforms it back to physical space, and
     * reports max-abs / RMS differences vs the {@code wxRef/wyRef/wzRef}
     * reference triples (typically the vorticity columns from a CSV
     * checkpoint). Never blocks the run — a mismatch is logged, not thrown.
     */
    private void compareVorticityAgainstCurl(double[] wxRef, double[] wyRef, double[] wzRef) {
        double[] uxRe = veloXRe.toDoubleArray();
        double[] uxIm = veloXIm.toDoubleArray();
        double[] uyRe = veloYRe.toDoubleArray();
        double[] uyIm = veloYIm.toDoubleArray();
        double[] uzRe = veloZRe.toDoubleArray();
        double[] uzIm = veloZIm.toDoubleArray();

        int specSize = n1 * n2 * half;
        double[] wxReSpec = new double[specSize];
        double[] wxImSpec = new double[specSize];
        double[] wyReSpec = new double[specSize];
        double[] wyImSpec = new double[specSize];
        double[] wzReSpec = new double[specSize];
        double[] wzImSpec = new double[specSize];

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                for (int k = 0; k < half; k++) {
                    int idx = (i * n2 + j) * half + k;
                    double kxv = kx[i], kyv = ky[j], kzv = kz[k];
                    // ω̂ = i k × û  ⇒  ω̂.re = -k×û_im, ω̂.im = k×û_re
                    wxReSpec[idx] = -(kyv * uzIm[idx] - kzv * uyIm[idx]);
                    wxImSpec[idx] =  (kyv * uzRe[idx] - kzv * uyRe[idx]);
                    wyReSpec[idx] = -(kzv * uxIm[idx] - kxv * uzIm[idx]);
                    wyImSpec[idx] =  (kzv * uxRe[idx] - kxv * uzRe[idx]);
                    wzReSpec[idx] = -(kxv * uyIm[idx] - kyv * uxIm[idx]);
                    wzImSpec[idx] =  (kxv * uyRe[idx] - kyv * uxRe[idx]);
                }
            }
        }

        Shape sShape = new Shape(n1, n2, half);
        double[] wxCurl = Fft.irfft3(wrap(wxReSpec, sShape), wrap(wxImSpec, sShape)).toDoubleArray();
        double[] wyCurl = Fft.irfft3(wrap(wyReSpec, sShape), wrap(wyImSpec, sShape)).toDoubleArray();
        double[] wzCurl = Fft.irfft3(wrap(wzReSpec, sShape), wrap(wzImSpec, sShape)).toDoubleArray();

        double maxAbsX = 0, maxAbsY = 0, maxAbsZ = 0;
        double sumSqX  = 0, sumSqY  = 0, sumSqZ  = 0;
        double refMaxAbs = 0, refSumSq = 0;
        int total = wxRef.length;
        for (int i = 0; i < total; i++) {
            double dx = wxCurl[i] - wxRef[i];
            double dy = wyCurl[i] - wyRef[i];
            double dz = wzCurl[i] - wzRef[i];
            maxAbsX = Math.max(maxAbsX, Math.abs(dx));
            maxAbsY = Math.max(maxAbsY, Math.abs(dy));
            maxAbsZ = Math.max(maxAbsZ, Math.abs(dz));
            sumSqX += dx*dx; sumSqY += dy*dy; sumSqZ += dz*dz;
            double refMag = Math.abs(wxRef[i]) + Math.abs(wyRef[i]) + Math.abs(wzRef[i]);
            refMaxAbs = Math.max(refMaxAbs, refMag);
            refSumSq += wxRef[i]*wxRef[i] + wyRef[i]*wyRef[i] + wzRef[i]*wzRef[i];
        }
        double rmsX = Math.sqrt(sumSqX / total);
        double rmsY = Math.sqrt(sumSqY / total);
        double rmsZ = Math.sqrt(sumSqZ / total);
        double refRms = Math.sqrt(refSumSq / total);
        double relRms = refRms > 0 ? Math.sqrt((sumSqX + sumSqY + sumSqZ) / total) / refRms : Double.NaN;

        double tol = 1e-6;
        String severity = (relRms > tol) ? "WARN" : "OK";
        logger.info("Vorticity check ({}): max|Δω|=({}, {}, {})  RMS=({}, {}, {})  rel-RMS={}  (ref maxAbs={}, RMS={})",
            severity, maxAbsX, maxAbsY, maxAbsZ, rmsX, rmsY, rmsZ, relRms, refMaxAbs, refRms);
        if (relRms > tol) {
            logger.warn("Vorticity from CSV disagrees with curl(u) by rel-RMS={} > tol={}. Continuing.", relRms, tol);
        }
    }

    /**
     * Pretty-prints every effective setting at simulation start:
     * grid + derived box parameters, the value of every knob that
     * {@link #loadProperties} may have overridden, and the two runtime
     * switches ({@link #device}, {@link #randomSeed}) that come from
     * {@code main()}'s env vars. Grouped by concern and framed with a
     * banner so it's easy to skim in a long log.
     *
     * <p>Prints defaults verbatim — the log shows what the run WILL
     * use, not a diff against defaults. That way a run's log is
     * self-describing and can be replayed from what it printed.
     */
    private void logSettings() {
        String bar = "════════════════════════════════════════════════════════════════════════";
        logger.info(bar);
        logger.info("  NavierStokes3D — effective simulation settings");
        logger.info(bar);
        logger.info("  Grid            : {}×{}×{}   (half = {} Fourier-half along axis-3)",
                    n1, n2, n3, half);
        logger.info("  Box wavenumbers : fx={}  fy={}  fz={}", fx, fy, fz);
        logger.info("  Real spacing    : dx={}  dx2={}", dx, dx2);
        logger.info("  Device          : {}", device);
        logger.info("  Random seed     : {}   (override via HB4J_SEED)", randomSeed);
        logger.info("  Simulation ID   : {}   (SHA-256 short-hash of simulation.properties)", simulationID);
        logger.info("  Old-flow file   : {}", oldflowFile != null ? oldflowFile : "(none — fresh flow)");
        logger.info("  ── Time marching ─────────────────────────────────────────────────");
        logger.info("    nhalt (total steps)          = {}", nhalt);
        logger.info("    dt (step size)               = {}", dt);
        logger.info("    ttime (start time)           = {}", ttime);
        logger.info("    norder (stiffly-stable order)= {}", norder);
        logger.info("    firstOrder (force order 1?)  = {}", firstOrder);
        logger.info("  ── Logging & I/O ────────────────────────────────────────────────");
        logger.info("    nshort (stats every N steps) = {}", nshort);
        logger.info("    nspec  (spectra every N stps)= {}", nspec);
        logger.info("    storeFile (write checkpoint) = {}", storeFile);
        logger.info("  ── De-aliasing & viscosity ──────────────────────────────────────");
        logger.info("    cutoff (spectral cutoff)     = {}", cutoff);
        logger.info("    rnu    (kinematic viscosity) = {}", rnu);
        logger.info("    filterRhalf                  = {}", filterRhalf);
        logger.info("  ── Initial flow ─────────────────────────────────────────────────");
        logger.info("    newflo (fresh vs. restart)   = {}", newflo);
        logger.info("    initialFlow (1=random,2=vortex,3=tube-along-z,4=two-parallel-tubes)= {}", initialFlow);
        logger.info("    scaleFlow → E={} per component", scaleFlow ? String.valueOf(scaleEnergy) : "OFF");
        logger.info("    facp (peak wavenumber)       = {}", facp);
        logger.info("    plank (spectrum shape)       = {}", plank);
        logger.info("    scaleflow / traveling        = {} / {}", scaleFlow, traveling);
        logger.info("    c1..c4 (spectrum constants)  = {}, {}, {}, {}", c1, c2, c3, c4);
        logger.info("  ── Vortex-cell params (used only if initialFlow=2) ─────────────");
        logger.info("    vAlpha={}  vBeta={}  strength={}  tubeSep={}",
            java.util.Arrays.toString(vAlpha),
            java.util.Arrays.toString(vBeta),
            java.util.Arrays.toString(vortexStrength),
            vortexTubeSeparation);
        logger.info("    vCell={}   vStep={}   vRatio={}", vCell, vStep, vRatio);
        logger.info("  ── Forcing ──────────────────────────────────────────────────────");
        logger.info("    uForce = {}", uForce);
        logger.info("  ── Assumptions ──────────────────────────────────────────────────");
        logger.info("    FFT convention: forward unscaled, inverse ×1/N (numpy style)");
        logger.info("    Periodic cubic domain of side 2π");
        logger.info("    Marching: implicit diffusion, explicit projection for div-free");
        logger.info("    Stats files: ShortStats.dat, Spectra.dat (raw magnitudes)");
        logger.info(bar);
    }

    /**
     * Initializes the simulation arrays and loads either the configured initial flow
     * or a checkpoint for a restart run.
     */
    public void initialize() {
        logSettings();
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
            } else if (initialFlow == 3) {
                FlowFieldInitializer.initializeVortexTube(this);
            } else if (initialFlow == 4) {
                FlowFieldInitializer.initializeTwoVortexTubes(this);
            }
            // ranflow sets Fourier coefficients on an O(0.1) amplitude
            // scale, but with our numpy-style FFT normalization
            // (forward unscaled, inverse ×1/N) this yields physical
            // velocities of O(1/N), i.e., energy O(1/N²) — machine
            // epsilon at typical grid sizes. Renormalize the whole
            // spectrum here so the initial physical energy equals the
            // configured target. The Fortran reference declared this
            // toggle but never actually implemented the rescale.
            if (scaleFlow) {
                FlowFieldInitializer.rescaleInitialEnergy(this, scaleEnergy);
            }
        } else {
            // Restart path: newflo=false MUST come with an explicit
            // checkpoint file, either via simulation.oldflow.file or
            // main()'s positional arg. Silently defaulting to some
            // filename in the cwd is a nice way to load a stale
            // "checkpoint.dat" from an unrelated run.
            if (oldflowFile == null) {
                throw new IllegalStateException(
                    "simulation.newflow=false requires an old-flow checkpoint. " +
                    "Set 'simulation.oldflow.file=<path>' in the properties file, " +
                    "or pass the path as the first positional arg to main().");
            }
            logger.info("Restarting from checkpoint: {}", oldflowFile);
            loadCheckpoint(oldflowFile);
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
        // One tag per run — <simID>_<yyyyMMdd-HHmmss>. All three
        // artifacts written by this run (ShortStats, Spectra, checkpoint)
        // share it, so files from one simulation cluster together and
        // never collide with files from another run of the same config.
        // Timestamp is taken at run start (not end) so the ShortStats
        // and Spectra files can be tagged BEFORE their first append.
        String runStamp = new java.text.SimpleDateFormat("yyyyMMdd-HHmmss").format(new java.util.Date());
        String runTag = simulationID + "_" + runStamp;
        FlowStatistics.setOutputFiles(
            "ShortStats_" + runTag + ".dat",
            "Spectra_"    + runTag + ".dat");
        logger.info("Run tag: {}  → ShortStats_{}.dat, Spectra_{}.dat, checkpoint_{}.dat",
            runTag, runTag, runTag, runTag);

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

        // Per-iteration wall-time collected so we can report warmup (istep=0,
        // includes cuFFT plan build + JIT) separately from the steady-state
        // average. Enables host↔GPU benchmarking without external harness.
        long loopStartNs = System.nanoTime();
        long firstIterNs = 0L;

        for (int istep = 0; istep <= nhalt; istep++) {
            long iterStartNs = System.nanoTime();
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

            // Logging short statistics — pass velocity, vorticity, AND
            // Lamb vector so FlowStatistics can compute all four
            // physical invariants (energy, enstrophy, |Lamb|², helicity),
            // matching HB_lib2.f90 short_stat.
            if (istep % nshort == 0) {
                FlowStatistics.logShortStats(ttime, rnu, n1, n2, n3,
                    uData, vData, wData,
                    wxData, wyData, wzData,
                    LxData, LyData, LzData);
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
                FlowStatistics.computeAndWriteSpectra(FlowStatistics.SPECTRA_FILE,
                                                      rnu, ttime, n1, n2, n3, half,
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
            long iterNs = System.nanoTime() - iterStartNs;
            if (istep == 0) firstIterNs = iterNs;
        }

        long totalNs = System.nanoTime() - loopStartNs;
        long steadyNs = totalNs - firstIterNs;
        int steadyIters = Math.max(nhalt, 1); // istep=1..nhalt
        logger.info("Simulation loop finished.");
        logger.info(String.format(
            "Timing on %s: %d steps in %.2f s (first %.2f s, steady-state %.3f s/iter avg)",
            device.getName(), nhalt + 1, totalNs / 1e9,
            firstIterNs / 1e9, (steadyNs / 1e9) / steadyIters));
        if (storeFile) {
            // Same runTag as the stats/spectra files above — the three
            // artifacts from one run all share the same identifier.
            saveCheckpoint("checkpoint_" + runTag + ".dat");
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
    public int getN3() { return n3; }
    public double getRnu() { return rnu; }
    public double getVAlpha() { return vAlpha[0]; }
    public double getVBeta() { return vBeta[0]; }
    public double getVortexStrength() { return vortexStrength[0]; }
    public double[] getVAlphaArray() { return vAlpha; }
    public double[] getVBetaArray() { return vBeta; }
    public double[] getVortexStrengthArray() { return vortexStrength; }
    public double getVortexTubeSeparation() { return vortexTubeSeparation; }
    public int getVCell() { return vCell; }
    public int getVStep() { return vStep; }
    public double getVRatio() { return vRatio; }
    public long getRandomSeed() { return randomSeed; }
    public void setRandomSeed(long seed) { this.randomSeed = seed; }

    public NDArray getVeloXRe() { return veloXRe; }
    public NDArray getVeloXIm() { return veloXIm; }
    public NDArray getVeloYRe() { return veloYRe; }
    public NDArray getVeloYIm() { return veloYIm; }
    public NDArray getVeloZRe() { return veloZRe; }
    public NDArray getVeloZIm() { return veloZIm; }

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
        String oldflowArg = null;
        for (String a : args) {
            if ("--gpu".equalsIgnoreCase(a)) {
                useGpu = true;
            } else if (!a.startsWith("-")) {
                // First non-flag positional arg is the restart checkpoint
                // path (used when simulation.newflow=false and the
                // properties file doesn't set simulation.oldflow.file).
                if (oldflowArg == null) oldflowArg = a;
            }
        }
        String envDev = System.getenv("HB4J_DEVICE");
        if (envDev != null && envDev.equalsIgnoreCase("gpu")) useGpu = true;

        // Grid size and iteration count are env-configurable so the
        // benchmark can compare 64³/128³ host vs GPU without touching
        // source or the properties file.
        int grid = envIntOr("HB4J_GRID", 64);
        Integer nhaltOverride = envIntOrNull("HB4J_NHALT");
        Long seedOverride = envLongOrNull("HB4J_SEED");

        NavierStokes3D sim = new NavierStokes3D(grid, grid, grid);
        if (useGpu) {
            Device d = Device.defaultDevice();
            sim.setDevice(d);
            logger.info("GPU mode requested: dispatching FFTs on {}", d);
        } else {
            logger.info("Host mode (pass --gpu to route FFTs through cuFFT)");
        }
        sim.loadProperties("src/test/resources/simulation.properties");
        if (nhaltOverride != null) sim.nhalt = nhaltOverride;
        if (seedOverride != null) sim.setRandomSeed(seedOverride);
        if (oldflowArg != null) sim.setOldflowFile(oldflowArg);
        logger.info("Grid = {}³, nhalt = {}, seed = {}", grid, sim.nhalt, sim.getRandomSeed());
        sim.initialize();
        sim.run();
    }
}
