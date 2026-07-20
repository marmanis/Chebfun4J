package com.marmanis.chebfun4j.examples.hb4j;

import com.marmanis.jax4j.core.NDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Color;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Responsible for computing, logging, and plotting physical and statistical properties
 * of the 3-D fluid velocity fields.
 */
public class FlowStatistics {
    private static final Logger logger = LoggerFactory.getLogger(FlowStatistics.class);

    // Buffer to collect data points for visual plotting
    private static final List<Double> timeList = new ArrayList<>();
    private static final List<Double> a1List = new ArrayList<>();
    private static final List<Double> a2List = new ArrayList<>();
    private static final List<Double> a3List = new ArrayList<>();

    private static double[] lastWavenumbers;
    private static double[] lastPhi1;
    private static double[] lastPhi2;
    private static double[] lastPhi3;
    private static double[] lastPhi4;

    /**
     * Computes and logs the short statistics of the simulation including kinetic energy
     * components and isotropy index values.
     *
     * @param ttime current simulation time
     * @param n1 grid resolution along the x-axis
     * @param n2 grid resolution along the y-axis
     * @param n3 grid resolution along the z-axis
     * @param u physical velocity field component in the x-direction
     * @param v physical velocity field component in the y-direction
     * @param w physical velocity field component in the z-direction
     */
    public static void logShortStats(double ttime, int n1, int n2, int n3, double[] u, double[] v, double[] w) {
        int npts = n1 * n2 * n3;
        double sumU2 = 0.0, sumV2 = 0.0, sumW2 = 0.0;

        for (int i = 0; i < npts; i++) {
            sumU2 += u[i] * u[i];
            sumV2 += v[i] * v[i];
            sumW2 += w[i] * w[i];
        }

        double probe1 = sumU2 / npts;
        double probe2 = sumV2 / npts;
        double probe3 = sumW2 / npts;
        double totalE = probe1 + probe2 + probe3;

        double a1 = totalE > 1e-15 ? Math.abs(probe1 / totalE) : 0.0;
        double a2 = totalE > 1e-15 ? Math.abs(probe2 / totalE) : 0.0;
        double a3 = totalE > 1e-15 ? Math.abs(probe3 / totalE) : 0.0;

        logger.info(String.format("%10.6f %10.6f %10.6f %10.6f", ttime, a1, a2, a3));

        // Save to buffers for plotting
        timeList.add(ttime);
        a1List.add(a1);
        a2List.add(a2);
        a3List.add(a3);
    }

    /**
     * Computes 3-D spherical shell spectra for energy, dissipation, helicity-like u.Lamb, and
     * Lamb vector energy, and writes the normalized results to a Spectra file.
     */
    public static void computeAndWriteSpectra(String filename, double rnu, double ttime, int n1, int n2, int n3, int half,
                                              double[] kx2, double[] ky2, double[] kz2,
                                              NDArray veloXRe, NDArray veloXIm,
                                              NDArray veloYRe, NDArray veloYIm,
                                              NDArray veloZRe, NDArray veloZIm,
                                              NDArray LxRe, NDArray LxIm,
                                              NDArray LyRe, NDArray LyIm,
                                              NDArray LzRe, NDArray LzIm) {
        int n1h = n1 / 2;
        double[] phi1 = new double[n1h + 1];
        double[] phi2 = new double[n1h + 1];
        double[] phi3 = new double[n1h + 1];
        double[] phi4 = new double[n1h + 1];

        double[] vxRe = veloXRe.toDoubleArray();
        double[] vxIm = veloXIm.toDoubleArray();
        double[] vyRe = veloYRe.toDoubleArray();
        double[] vyIm = veloYIm.toDoubleArray();
        double[] vzRe = veloZRe.toDoubleArray();
        double[] vzIm = veloZIm.toDoubleArray();

        double[] lxRe = LxRe.toDoubleArray();
        double[] lxIm = LxIm.toDoubleArray();
        double[] lyRe = LyRe.toDoubleArray();
        double[] lyIm = LyIm.toDoubleArray();
        double[] lzRe = LzRe.toDoubleArray();
        double[] lzIm = LzIm.toDoubleArray();

        for (int i = 0; i < n1; i++) {
            double kxVal2 = kx2[i];
            for (int j = 0; j < n2; j++) {
                double kyVal2 = ky2[j];
                for (int k = 0; k < half; k++) {
                    double kzVal2 = kz2[k];
                    double ktemp = Math.sqrt(kxVal2 + kyVal2 + kzVal2);
                    int itemp = (int) ktemp;

                    if (itemp > 0 && itemp <= n1h) {
                        double xmult = (k == 0 || k == half - 1) ? 1.0 : 2.0;
                        int idx = (i * n2 + j) * half + k;

                        double usqr = xmult * (vxRe[idx] * vxRe[idx] + vxIm[idx] * vxIm[idx] +
                                               vyRe[idx] * vyRe[idx] + vyIm[idx] * vyIm[idx] +
                                               vzRe[idx] * vzRe[idx] + vzIm[idx] * vzIm[idx]);

                        double uLambx = xmult * (vxRe[idx] * lxRe[idx] + vxIm[idx] * lxIm[idx] +
                                                 vyRe[idx] * lyRe[idx] + vyIm[idx] * lyIm[idx] +
                                                 vzRe[idx] * lzRe[idx] + vzIm[idx] * lzIm[idx]);

                        double slamb = xmult * (lxRe[idx] * lxRe[idx] + lxIm[idx] * lxIm[idx] +
                                                lyRe[idx] * lyRe[idx] + lyIm[idx] * lyIm[idx] +
                                                lzRe[idx] * lzRe[idx] + lzIm[idx] * lzIm[idx]);

                        phi1[itemp] += usqr;
                        phi2[itemp] += usqr * (kxVal2 + kyVal2 + kzVal2);
                        phi3[itemp] += uLambx;
                        phi4[itemp] += slamb;
                    }
                }
            }
        }

        for (int i = 0; i <= n1h; i++) {
            phi2[i] = rnu * phi2[i];
        }

        double max_phi1 = 1e-15, max_phi2 = 1e-15, max_phi3 = 1e-15, max_phi4 = 1e-15;
        for (int i = 1; i <= n1h; i++) {
            max_phi1 = Math.max(max_phi1, Math.abs(phi1[i]));
            max_phi2 = Math.max(max_phi2, Math.abs(phi2[i]));
            max_phi3 = Math.max(max_phi3, Math.abs(phi3[i]));
            max_phi4 = Math.max(max_phi4, Math.abs(phi4[i]));
        }

        // Save normalized spectra to static buffers for plotting
        lastWavenumbers = new double[n1h];
        lastPhi1 = new double[n1h];
        lastPhi2 = new double[n1h];
        lastPhi3 = new double[n1h];
        lastPhi4 = new double[n1h];
        for (int i = 1; i <= n1h; i++) {
            lastWavenumbers[i - 1] = i;
            lastPhi1[i - 1] = (phi1[i] == 0.0) ? 1.0e-12 : phi1[i] / max_phi1;
            lastPhi2[i - 1] = (phi2[i] == 0.0) ? 1.0e-12 : phi2[i] / max_phi2;
            lastPhi3[i - 1] = (phi3[i] == 0.0) ? 1.0e-12 : phi3[i] / max_phi3;
            lastPhi4[i - 1] = (phi4[i] <= 0.0) ? 1.0e-12 : phi4[i] / max_phi4;
        }

        boolean append = new File(filename).exists() && ttime > 0.0;
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(filename, append)))) {
            if (!append) {
                writer.println("TITLE = \"Energy Spectra\"");
            }
            writer.printf("ZONE I=%d%n", n1h);
            for (int i = 1; i <= n1h; i++) {
                double xt1 = (phi1[i] == 0.0) ? 1.0e-12 : phi1[i] / max_phi1;
                double xt2 = (phi2[i] == 0.0) ? 1.0e-12 : phi2[i] / max_phi2;
                double xt3 = (phi3[i] == 0.0) ? 1.0e-12 : phi3[i] / max_phi3;
                double xt4 = (phi4[i] <= 0.0) ? 1.0e-12 : phi4[i] / max_phi4;
                writer.printf(" %5d   %12.5e   %12.5e   %12.5e   %12.5e%n", i, xt1, xt2, xt3, xt4);
            }
            logger.info("Spectra written successfully to: {}", filename);
        } catch (IOException e) {
            logger.error("Error writing spectra: {}", e.getMessage(), e);
        }
    }

    /**
     * Renders a Swing window displaying both short statistics and spectra using jMatplot.
     */
    public static void showPlots() {
        if (java.awt.GraphicsEnvironment.isHeadless()) {
            logger.info("Headless environment detected. Skipping interactive Swing plot rendering.");
            return;
        }

        if (timeList.isEmpty()) {
            logger.warn("No statistical data collected to plot.");
            return;
        }

        double[] time = timeList.stream().mapToDouble(Double::doubleValue).toArray();
        double[] a1 = a1List.stream().mapToDouble(Double::doubleValue).toArray();
        double[] a2 = a2List.stream().mapToDouble(Double::doubleValue).toArray();
        double[] a3 = a3List.stream().mapToDouble(Double::doubleValue).toArray();

        // 1. Create short stats Figure
        com.marmanis.jMatplot.core.Figure figStats = new com.marmanis.jMatplot.core.Figure();
        com.marmanis.jMatplot.core.Axes axStats = figStats.addAxes();

        com.marmanis.jMatplot.core.Line2D l1 = axStats.plot(time, a1);
        l1.setColor(Color.RED);
        l1.setLabel("a1 (u^2 fraction)");

        com.marmanis.jMatplot.core.Line2D l2 = axStats.plot(time, a2);
        l2.setColor(Color.GREEN);
        l2.setLabel("a2 (v^2 fraction)");

        com.marmanis.jMatplot.core.Line2D l3 = axStats.plot(time, a3);
        l3.setColor(Color.BLUE);
        l3.setLabel("a3 (w^2 fraction)");

        axStats.setXLabel("Time");
        axStats.setYLabel("Fraction of Energy");
        axStats.setTitle("Short Statistics: Energy Components");
        axStats.legend();

        // 2. Create spectra Figure
        com.marmanis.jMatplot.core.Figure figSpectra = new com.marmanis.jMatplot.core.Figure();
        com.marmanis.jMatplot.core.Axes axSpectra = figSpectra.addAxes();
        if (lastWavenumbers != null) {
            com.marmanis.jMatplot.core.Line2D s1 = axSpectra.plot(lastWavenumbers, lastPhi1);
            s1.setColor(Color.RED);
            s1.setLabel("Energy Spectrum (phi1)");

            com.marmanis.jMatplot.core.Line2D s2 = axSpectra.plot(lastWavenumbers, lastPhi2);
            s2.setColor(Color.GREEN);
            s2.setLabel("Dissipation Spectrum (phi2)");

            com.marmanis.jMatplot.core.Line2D s3 = axSpectra.plot(lastWavenumbers, lastPhi3);
            s3.setColor(Color.BLUE);
            s3.setLabel("Helicity Spectrum (phi3)");

            com.marmanis.jMatplot.core.Line2D s4 = axSpectra.plot(lastWavenumbers, lastPhi4);
            s4.setColor(Color.ORANGE);
            s4.setLabel("Lamb Vector Spectrum (phi4)");

            axSpectra.setXLabel("Wavenumber k");
            axSpectra.setYLabel("Normalized Spectrum Value");
            axSpectra.setTitle("Flow Spectra");
            axSpectra.legend();
        } else {
            axSpectra.setTitle("No Spectra Data Available");
        }

        // 3. Create components and Swing window
        com.marmanis.jMatplot.core.PlotPanel panelStats = new com.marmanis.jMatplot.core.PlotPanel(figStats);
        com.marmanis.jMatplot.core.PlotPanel panelSpectra = new com.marmanis.jMatplot.core.PlotPanel(figSpectra);

        javax.swing.JFrame frame = new javax.swing.JFrame("Navier-Stokes 3-D Simulation Analysis");
        frame.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(800, 600);

        javax.swing.JTabbedPane tabbedPane = new javax.swing.JTabbedPane();
        tabbedPane.addTab("Short Statistics", panelStats);
        tabbedPane.addTab("Flow Spectra", panelSpectra);

        frame.getContentPane().add(tabbedPane);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        logger.info("Simulation analysis plots window displayed successfully.");
    }
}
