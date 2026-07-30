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

    // File-based data flow: the simulation appends one row per stats step
    // to SHORT_STATS_FILE, and one block per spectra step to SPECTRA_FILE.
    // At the end of the run, showPlots re-reads both files from scratch
    // and builds the plots — no in-memory carry-over. This decouples
    // compute from visualization and lets us open the plot window later
    // (or from a separate program) against any completed run's artifacts.
    // Output filenames — mutable so a driver (NavierStokes3D main) can
    // tag each run's artifacts with the same <simID>_<timestamp> the
    // checkpoint uses, so all three files from one run cluster
    // together on disk. Defaults kept for PlotViewer's ergonomic path.
    public static String SHORT_STATS_FILE = "ShortStats.dat";
    public static String SPECTRA_FILE     = "Spectra.dat";

    /**
     * Redirect subsequent {@code logShortStats} / {@code computeAndWriteSpectra}
     * output. Must be called BEFORE the first stats step, otherwise
     * the header row lands in the old file. Passing {@code null}
     * leaves that name at its current value.
     */
    public static void setOutputFiles(String shortStatsPath, String spectraPath) {
        if (shortStatsPath != null) SHORT_STATS_FILE = shortStatsPath;
        if (spectraPath    != null) SPECTRA_FILE     = spectraPath;
        // Force a fresh file on next append — the previous filename's
        // "initialized" flag would otherwise skip the header.
        shortStatsFileInitialized = false;
        spectraFileInitialized    = false;
    }

    // Tracks whether we've truncated the output files this JVM. The
    // first call to log/compute per file gets a fresh file; subsequent
    // calls append. This matches the run-lifecycle better than the
    // ttime==0 heuristic previously used, which fails on restart.
    private static boolean shortStatsFileInitialized = false;
    private static boolean spectraFileInitialized    = false;

    /**
     * One row of short-stats output as loaded from disk.
     * All four quantities are per grid cell (divided by n1*n2*n3),
     * matching the Fortran short_stat convention.
     */
    private record ShortStatsRow(
        double ttime,
        double energy,       // Σ|u|²   / N   -- probe(9)
        double enstrophy,    // Σ|ω|²   / N   -- probe(4+5+6)
        double dissipation,  // ν · enstrophy  -- probe(8)
        double lambSq,       // Σ|u×ω|² / N   -- probe(10), a.k.a. strophokinesis
        double helicity) {}  // Σ u·ω   / N   -- probe(7)

    /**
     * One 1-D spherical-shell spectrum snapshot at a fixed simulation
     * time. All four φ arrays share {@link #wavenumbers} as their
     * x-axis and are stored in RAW magnitudes (not per-snapshot
     * normalized), so decay across time is visible when scrubbing.
     */
    private record SpectraSnapshot(
        double ttime,
        double[] wavenumbers,
        double[] phi1,
        double[] phi2,
        double[] phi3,
        double[] phi4) {}

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
    /**
     * Computes the four physical invariants tracked by the Fortran
     * reference short_stat: total kinetic energy, enstrophy, squared
     * Lamb vector, and helicity — all per grid cell.
     *
     * <p>Inputs are the physical-space (already-inverted) velocity,
     * vorticity, and Lamb vector — the caller already has all three
     * because they were just computed for the marching scheme. The
     * three isotropy fractions {@code a1/a2/a3} are logged for
     * quick-eyeball but not stored (the four scalars above are what
     * the animation panel plots).
     */
    public static void logShortStats(
            double ttime, double rnu, int n1, int n2, int n3,
            double[] u,  double[] v,  double[] w,
            double[] wx, double[] wy, double[] wz,
            double[] Lx, double[] Ly, double[] Lz) {
        int npts = n1 * n2 * n3;
        double sumU2 = 0.0, sumV2 = 0.0, sumW2 = 0.0;
        double sumWx2 = 0.0, sumWy2 = 0.0, sumWz2 = 0.0;
        double sumHel = 0.0, sumLambSq = 0.0;

        for (int i = 0; i < npts; i++) {
            sumU2  += u[i]  * u[i];
            sumV2  += v[i]  * v[i];
            sumW2  += w[i]  * w[i];
            sumWx2 += wx[i] * wx[i];
            sumWy2 += wy[i] * wy[i];
            sumWz2 += wz[i] * wz[i];
            sumHel += u[i] * wx[i] + v[i] * wy[i] + w[i] * wz[i];
            sumLambSq += Lx[i] * Lx[i] + Ly[i] * Ly[i] + Lz[i] * Lz[i];
        }

        double energy      = (sumU2 + sumV2 + sumW2)   / npts;   // probe(9)
        double enstrophy   = (sumWx2 + sumWy2 + sumWz2) / npts;  // probe(4+5+6)
        double dissipation = rnu * enstrophy;                    // probe(8) = ν·Ω
        double lambSq      = sumLambSq / npts;                   // probe(10)
        double helicity    = sumHel / npts;                      // probe(7)

        double a1 = energy > 1e-15 ? Math.abs((sumU2 / npts) / energy) : 0.0;
        double a2 = energy > 1e-15 ? Math.abs((sumV2 / npts) / energy) : 0.0;
        double a3 = energy > 1e-15 ? Math.abs((sumW2 / npts) / energy) : 0.0;

        logger.info(String.format(
            "t=%10.6f  E=%12.5e  Ω=%12.5e  ε=%12.5e  |L|²=%12.5e  H=%12.5e  (a1=%.3f a2=%.3f a3=%.3f)",
            ttime, energy, enstrophy, dissipation, lambSq, helicity, a1, a2, a3));

        appendShortStatsRow(ttime, energy, enstrophy, dissipation, lambSq, helicity);
    }

    /**
     * Appends one row to {@link #SHORT_STATS_FILE}. First call this JVM
     * truncates the file and writes a header; subsequent calls append.
     * Format is space-separated fixed-width so it's trivially readable
     * by other tools (numpy loadtxt, awk, etc.).
     */
    private static void appendShortStatsRow(double ttime, double energy,
                                            double enstrophy, double dissipation,
                                            double lambSq, double helicity) {
        boolean append = shortStatsFileInitialized;
        try (PrintWriter w = new PrintWriter(new BufferedWriter(new FileWriter(SHORT_STATS_FILE, append)))) {
            if (!append) {
                w.println("# HB4J short-stats — one row per stats step (per-cell, /N)");
                w.println("# columns: ttime  energy  enstrophy  dissipation(ν·Ω)  lambSq(strophokinesis)  helicity");
                shortStatsFileInitialized = true;
            }
            w.printf("%14.7e  %14.7e  %14.7e  %14.7e  %14.7e  %14.7e%n",
                ttime, energy, enstrophy, dissipation, lambSq, helicity);
        } catch (IOException e) {
            logger.error("Failed to write short stats to {}: {}", SHORT_STATS_FILE, e.getMessage());
        }
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

        // Write raw magnitudes to disk. showPlots normalizes globally
        // across all snapshots at plot time (see loadSpectraNormalized)
        // so the curves stay comparable across timesteps without needing
        // log-y in the plot — which was causing jMatplot to hang.
        boolean append = spectraFileInitialized;
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(filename, append)))) {
            if (!append) {
                writer.println("# HB4J spherical-shell spectra — one row per (time, k) tuple, raw magnitudes");
                writer.println("# columns: ttime  k  phi1(energy)  phi2(dissipation)  phi3(helicity)  phi4(|Lamb|²)");
                spectraFileInitialized = true;
            }
            for (int i = 1; i <= n1h; i++) {
                // Guard against NaN/Inf leaking into the plot pipeline.
                // Negative helicity values are fine — they're stored as-is.
                double p1 = safeFinite(phi1[i]);
                double p2 = safeFinite(phi2[i]);
                double p3 = safeFinite(phi3[i]);
                double p4 = safeFinite(phi4[i]);
                writer.printf("%14.7e  %5d  %14.7e  %14.7e  %14.7e  %14.7e%n",
                    ttime, i, p1, p2, p3, p4);
            }
        } catch (IOException e) {
            logger.error("Error writing spectra to {}: {}", filename, e.getMessage(), e);
        }
    }

    private static double safeFinite(double v) {
        return Double.isFinite(v) ? v : 0.0;
    }

    /**
     * Convenience: render using the default filenames ({@link #SHORT_STATS_FILE},
     * {@link #SPECTRA_FILE}) in the current working directory.
     */
    public static void showPlots() {
        showPlots(SHORT_STATS_FILE, SPECTRA_FILE);
    }

    /**
     * Renders a Swing window displaying both short statistics and spectra
     * using jMatplot, loading the raw data from the two named files.
     * Accepts arbitrary paths so multiple completed runs can be compared
     * by opening the same viewer against different artifact sets.
     *
     * @param shortStatsPath path to a ShortStats.dat-format file
     * @param spectraPath    path to a Spectra.dat-format file
     */
    public static void showPlots(String shortStatsPath, String spectraPath) {
        if (java.awt.GraphicsEnvironment.isHeadless()) {
            logger.info("Headless environment detected. Skipping interactive Swing plot rendering.");
            return;
        }

        // Load everything from disk — the simulation appended rows to
        // the two files as it ran. Reading here means the whole plot
        // subsystem is decoupled from the sim's in-memory state, and
        // a completed run can be re-plotted later (or from an entirely
        // separate program) without rerunning.
        List<ShortStatsRow> stats;
        List<SpectraSnapshot> spectra;
        try {
            stats = readShortStats(shortStatsPath);
            spectra = readSpectra(spectraPath);
        } catch (IOException e) {
            logger.error("Failed to load plot data from disk: {}", e.getMessage(), e);
            return;
        }
        // Normalize each φ_i by its max across ALL snapshots (not per-
        // snapshot). Keeps every curve in [0, 1] for linear rendering
        // AND preserves relative magnitude between timesteps, so later
        // snapshots visibly droop as the flow decays.
        spectra = normalizeGlobally(spectra);
        if (stats.isEmpty()) {
            logger.warn("No stats rows in {} — skipping plot.", SHORT_STATS_FILE);
            return;
        }
        logger.info("Rendering plots: {} stats samples, {} spectra snapshots",
            stats.size(), spectra.size());

        double[] time      = stats.stream().mapToDouble(ShortStatsRow::ttime).toArray();
        double[] energy    = stats.stream().mapToDouble(ShortStatsRow::energy).toArray();
        double[] enstrophy = stats.stream().mapToDouble(ShortStatsRow::enstrophy).toArray();
        double[] lambSq    = stats.stream().mapToDouble(ShortStatsRow::lambSq).toArray();
        double[] helicity  = stats.stream().mapToDouble(ShortStatsRow::helicity).toArray();
        // dissipation column is still read + stored (for external tooling)
        // even though no in-tab plot consumes it after the Cascade tab
        // was removed.

        // Visibility state for each panel's curves — mutable arrays so
        // menu toggles change the flag in place and a refresh callback
        // reads the current value. All curves start visible.
        final boolean[] statsVisible = {true, true, true, true};
        final boolean[] specVisible  = {true, true, true, true};

        final double[] timeF = time, energyF = energy, enstrophyF = enstrophy,
                       lambSqF = lambSq, helicityF = helicity;
        final List<SpectraSnapshot> spectraFinal = spectra;

        // All Swing UI construction MUST run on the Event Dispatch
        // Thread. Building the JFrame on the caller's thread (main, or
        // the simulation loop thread) races the AWT paint dispatcher
        // and typically produces a window that paints once and then
        // stops responding to input — exactly what we were seeing.
        // invokeLater lets showPlots return immediately; the window's
        // lifecycle is managed by the EDT afterwards.
        javax.swing.SwingUtilities.invokeLater(() -> {
            com.marmanis.jMatplot.core.PlotPanel panelStats =
                new com.marmanis.jMatplot.core.PlotPanel(
                    buildStatsFigure(timeF, energyF, enstrophyF, lambSqF, helicityF, statsVisible));
            SpectraTabView spectraTabView = buildSpectraTab(spectraFinal, specVisible);

            Runnable refreshStats = () -> panelStats.setFigure(
                buildStatsFigure(timeF, energyF, enstrophyF, lambSqF, helicityF, statsVisible));

            javax.swing.JFrame frame = new javax.swing.JFrame("Navier-Stokes 3-D Simulation Analysis");
            frame.setDefaultCloseOperation(javax.swing.JFrame.DISPOSE_ON_CLOSE);
            frame.setSize(800, 600);
            frame.setJMenuBar(buildViewMenuBar(statsVisible, specVisible, refreshStats, spectraTabView.refresh()));

            javax.swing.JTabbedPane tabbedPane = new javax.swing.JTabbedPane();
            tabbedPane.addTab("Short Statistics", panelStats);
            tabbedPane.addTab("Flow Spectra", spectraTabView.tab());

            frame.getContentPane().add(tabbedPane);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
            logger.info("Simulation analysis plots window displayed successfully.");
        });
    }

    /** Curve-index labels for the stats visibility array (matches menu order). */
    static final String[] STATS_LABELS = {
        "Energy  Σ|u|²/N", "Enstrophy  Σ|ω|²/N", "|Lamb|²  Σ|u×ω|²/N", "Helicity  Σ(u·ω)/N"
    };
    private static final Color[] STATS_COLORS = { Color.RED, Color.GREEN, Color.BLUE, Color.ORANGE };

    /**
     * Builds the "Short Statistics" figure showing whichever of the
     * four physical invariants the visibility array marks true. Split
     * out from {@link #showPlots} so a menu toggle can rebuild it
     * with a different subset without recomputing anything.
     */
    private static com.marmanis.jMatplot.core.Figure buildStatsFigure(
            double[] time, double[] energy, double[] enstrophy, double[] lambSq, double[] helicity,
            boolean[] visible) {
        com.marmanis.jMatplot.core.Figure fig = new com.marmanis.jMatplot.core.Figure();
        com.marmanis.jMatplot.core.Axes ax = fig.addAxes();
        double[][] series = { energy, enstrophy, lambSq, helicity };
        for (int i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            com.marmanis.jMatplot.core.Line2D ln = ax.plot(time, series[i]);
            ln.setColor(STATS_COLORS[i]);
            ln.setLabel(STATS_LABELS[i]);
        }
        ax.setXLabel("Time");
        ax.setYLabel("Value (per cell)");
        ax.setTitle("Short Statistics: Energy · Enstrophy · Lamb² · Helicity");
        ax.legend();
        return fig;
    }

    /**
     * Assembles the frame's menu bar. Two "View" submenus — one per
     * plot panel — carry a checkbox per curve. Toggling a checkbox
     * flips the flag in the shared visibility array and calls the
     * relevant refresh callback, which rebuilds the figure and pushes
     * it to the PlotPanel via {@code setFigure}.
     */
    private static javax.swing.JMenuBar buildViewMenuBar(
            boolean[] statsVisible, boolean[] specVisible,
            Runnable refreshStats, Runnable refreshSpectra) {
        javax.swing.JMenuBar bar = new javax.swing.JMenuBar();
        javax.swing.JMenu view = new javax.swing.JMenu("View");

        javax.swing.JMenu statsMenu = new javax.swing.JMenu("Short Statistics curves");
        for (int i = 0; i < STATS_LABELS.length; i++) {
            final int idx = i;
            javax.swing.JCheckBoxMenuItem item = new javax.swing.JCheckBoxMenuItem(STATS_LABELS[i], statsVisible[i]);
            item.addActionListener(e -> {
                statsVisible[idx] = item.isSelected();
                refreshStats.run();
            });
            statsMenu.add(item);
        }
        javax.swing.JMenu specMenu = new javax.swing.JMenu("Flow Spectra curves");
        for (int i = 0; i < SPEC_LABELS.length; i++) {
            final int idx = i;
            javax.swing.JCheckBoxMenuItem item = new javax.swing.JCheckBoxMenuItem(SPEC_LABELS[i], specVisible[i]);
            item.addActionListener(e -> {
                specVisible[idx] = item.isSelected();
                refreshSpectra.run();
            });
            specMenu.add(item);
        }

        view.add(statsMenu);
        view.add(specMenu);
        bar.add(view);
        return bar;
    }

    /**
     * Parses {@link #SHORT_STATS_FILE}: comment lines (leading '#') are
     * skipped; every other line is expected to hold 5 whitespace-separated
     * doubles matching {@link ShortStatsRow}.
     */
    private static List<ShortStatsRow> readShortStats(String filename) throws IOException {
        List<ShortStatsRow> out = new ArrayList<>();
        File f = new File(filename);
        if (!f.exists()) return out;
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 6) continue;
                out.add(new ShortStatsRow(
                    Double.parseDouble(parts[0]),
                    Double.parseDouble(parts[1]),
                    Double.parseDouble(parts[2]),
                    Double.parseDouble(parts[3]),
                    Double.parseDouble(parts[4]),
                    Double.parseDouble(parts[5])));
            }
        }
        return out;
    }

    /**
     * Parses {@link #SPECTRA_FILE}: comment lines skipped; every data
     * line is (ttime, k, φ1, φ2, φ3, φ4). Consecutive rows sharing the
     * same ttime form one {@link SpectraSnapshot}. Assumes the file is
     * grouped by time (which the writer guarantees) so we can flush a
     * snapshot each time ttime changes.
     */
    private static List<SpectraSnapshot> readSpectra(String filename) throws IOException {
        List<SpectraSnapshot> out = new ArrayList<>();
        File f = new File(filename);
        if (!f.exists()) return out;
        List<Double> ks = new ArrayList<>();
        List<Double> p1 = new ArrayList<>();
        List<Double> p2 = new ArrayList<>();
        List<Double> p3 = new ArrayList<>();
        List<Double> p4 = new ArrayList<>();
        double curTime = Double.NaN;
        try (BufferedReader r = new BufferedReader(new FileReader(f))) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 6) continue;
                double t = Double.parseDouble(parts[0]);
                if (!Double.isNaN(curTime) && t != curTime && !ks.isEmpty()) {
                    out.add(flushSnapshot(curTime, ks, p1, p2, p3, p4));
                    ks.clear(); p1.clear(); p2.clear(); p3.clear(); p4.clear();
                }
                curTime = t;
                ks.add(Double.parseDouble(parts[1]));
                p1.add(Double.parseDouble(parts[2]));
                p2.add(Double.parseDouble(parts[3]));
                p3.add(Double.parseDouble(parts[4]));
                p4.add(Double.parseDouble(parts[5]));
            }
        }
        if (!ks.isEmpty()) out.add(flushSnapshot(curTime, ks, p1, p2, p3, p4));
        return out;
    }

    /**
     * Divides every φ_i array by the global max of that same φ_i taken
     * across ALL snapshots. Values that came in negative (φ_3 helicity)
     * or as sentinel zeros are handled by dividing by the max of the
     * absolute value; the sign is preserved.
     */
    private static List<SpectraSnapshot> normalizeGlobally(List<SpectraSnapshot> in) {
        if (in.isEmpty()) return in;
        double m1 = 0, m2 = 0, m3 = 0, m4 = 0;
        for (SpectraSnapshot s : in) {
            for (double v : s.phi1()) m1 = Math.max(m1, Math.abs(v));
            for (double v : s.phi2()) m2 = Math.max(m2, Math.abs(v));
            for (double v : s.phi3()) m3 = Math.max(m3, Math.abs(v));
            for (double v : s.phi4()) m4 = Math.max(m4, Math.abs(v));
        }
        // A zero max means every value was 0 — leave those arrays as-is.
        double d1 = m1 > 0 ? m1 : 1.0;
        double d2 = m2 > 0 ? m2 : 1.0;
        double d3 = m3 > 0 ? m3 : 1.0;
        double d4 = m4 > 0 ? m4 : 1.0;
        List<SpectraSnapshot> out = new ArrayList<>(in.size());
        for (SpectraSnapshot s : in) {
            out.add(new SpectraSnapshot(
                s.ttime(), s.wavenumbers(),
                divideBy(s.phi1(), d1),
                divideBy(s.phi2(), d2),
                divideBy(s.phi3(), d3),
                divideBy(s.phi4(), d4)));
        }
        return out;
    }

    private static double[] divideBy(double[] a, double d) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] / d;
        return r;
    }

    private static SpectraSnapshot flushSnapshot(
            double ttime, List<Double> k, List<Double> p1, List<Double> p2, List<Double> p3, List<Double> p4) {
        return new SpectraSnapshot(ttime,
            k.stream().mapToDouble(Double::doubleValue).toArray(),
            p1.stream().mapToDouble(Double::doubleValue).toArray(),
            p2.stream().mapToDouble(Double::doubleValue).toArray(),
            p3.stream().mapToDouble(Double::doubleValue).toArray(),
            p4.stream().mapToDouble(Double::doubleValue).toArray());
    }

    /**
     * Builds a Figure showing the four normalized spectra of a single
     * simulation-time snapshot. Rebuilt from scratch on every slider
     * tick — cheaper than mutating Line2D data and keeps the whole
     * render path stateless.
     */
    /** Curve-index labels for spectra visibility array (matches menu order). */
    static final String[] SPEC_LABELS = {
        "Energy (phi1)", "Dissipation (phi2)", "|Helicity| (|phi3|)", "Strophokinesis (phi4)"
    };
    private static final Color[] SPEC_COLORS = { Color.RED, Color.GREEN, Color.BLUE, Color.ORANGE };

    private static com.marmanis.jMatplot.core.Figure buildSpectraFigure(SpectraSnapshot s, boolean[] visible) {
        com.marmanis.jMatplot.core.Figure fig = new com.marmanis.jMatplot.core.Figure();
        com.marmanis.jMatplot.core.Axes ax = fig.addAxes();

        // Log-y needs strictly positive values. The load-path already
        // normalized every φ to [-1, 1] (global-max), so a floor of
        // 1e-6 is well below anything a healthy spectrum carries but
        // safely above zero. Take |·| so negative shell contributions
        // (φ3 helicity) show as magnitude; sign visibility isn't
        // useful on a log scale anyway.
        double[][] p = {
            absFloored(s.phi1(), Y_FLOOR),
            absFloored(s.phi2(), Y_FLOOR),
            absFloored(s.phi3(), Y_FLOOR),
            absFloored(s.phi4(), Y_FLOOR)
        };
        for (int i = 0; i < 4; i++) {
            if (!visible[i]) continue;
            com.marmanis.jMatplot.core.Line2D ln = ax.plot(s.wavenumbers(), p[i]);
            ln.setColor(SPEC_COLORS[i]);
            ln.setLabel(SPEC_LABELS[i]);
        }

        // X-axis stays fixed to the box's wavenumber range (1..N/2).
        // Y-axis: log-scale, pinned to [Y_FLOOR, 1.1] so the decade
        // ticks are identical for every snapshot — only the curves
        // move as the flow decays, not the axis scale.
        double kMin = s.wavenumbers()[0];
        double kMax = s.wavenumbers()[s.wavenumbers().length - 1];
        ax.setXLim(kMin, kMax);
        ax.setYLim(Y_FLOOR, 1.1);
        ax.setLogScale(false, true);
        ax.setLogYBase(10.0);
        ax.setShowMinorTicksY(true);
        ax.setShowMinorGridY(true);
        ax.setMinorGridStyleY("Dotted");
        ax.setXLabel("Wavenumber k");
        ax.setYLabel("Spectra (normalized to global Max, log₁₀)");
        ax.setTitle(String.format("Flow Spectra @ t = %.4f", s.ttime()));
        ax.legend();
        return fig;
    }

    /** Floor for log-y plotting — six decades below the pinned y-max of 1.1. */
    private static final double Y_FLOOR = 1e-4;

    private static double[] absFloored(double[] in, double floor) {
        double[] out = new double[in.length];
        for (int i = 0; i < in.length; i++) {
            double v = Math.abs(in[i]);
            out[i] = v > floor ? v : floor;
        }
        return out;
    }

    /**
     * Composes the animated spectra tab. Layout:
     * <pre>
     *   ┌─────────────────────────────────────────────┐
     *   │                                             │
     *   │           PlotPanel (Figure center)         │
     *   │                                             │
     *   ├─────────────────────────────────────────────┤
     *   │ [Play] ────────●─────────  t = 0.0123       │
     *   └─────────────────────────────────────────────┘
     * </pre>
     * Slider index maps 1:1 to the caller-supplied history. The Play button
     * drives a {@link javax.swing.Timer} that increments the slider once
     * per tick; the caller can pause at any point by clicking Pause, or
     * scrub manually and the animation resumes from wherever it lands.
     */
    /**
     * Holder for a spectra tab: the Swing component to insert plus a
     * {@link #refresh} callback the caller can invoke to redraw the
     * current snapshot after menu toggles change the visibility array.
     */
    private record SpectraTabView(javax.swing.JComponent tab, Runnable refresh) {}

    private static SpectraTabView buildSpectraTab(List<SpectraSnapshot> spectraHistory, boolean[] visible) {
        javax.swing.JPanel container = new javax.swing.JPanel(new java.awt.BorderLayout());

        if (spectraHistory.isEmpty()) {
            // No snapshots collected — render a message and no controls,
            // so the tab still exists and doesn't crash.
            com.marmanis.jMatplot.core.Figure fig = new com.marmanis.jMatplot.core.Figure();
            fig.addAxes().setTitle("No Spectra Data Available");
            container.add(new com.marmanis.jMatplot.core.PlotPanel(fig), java.awt.BorderLayout.CENTER);
            return new SpectraTabView(container, () -> {});
        }

        final int lastIdx = spectraHistory.size() - 1;
        com.marmanis.jMatplot.core.PlotPanel plotPanel =
            new com.marmanis.jMatplot.core.PlotPanel(buildSpectraFigure(spectraHistory.get(lastIdx), visible));

        // ── slider spans the whole recorded history, starting at the end ──
        javax.swing.JSlider slider = new javax.swing.JSlider(0, lastIdx, lastIdx);
        // Ticks: aim for ~10 major ticks regardless of history length.
        int major = Math.max(1, (lastIdx + 1) / 10);
        slider.setMajorTickSpacing(major);
        slider.setPaintTicks(true);

        javax.swing.JLabel timeLabel = new javax.swing.JLabel(
            String.format("t = %.4f", spectraHistory.get(lastIdx).ttime()));
        timeLabel.setBorder(javax.swing.BorderFactory.createEmptyBorder(0, 12, 0, 12));

        javax.swing.JButton playBtn = new javax.swing.JButton("▶ Play");

        // ── Timer drives auto-advance; ~5 fps keeps repaint cheap ──
        final int frameDelayMs = 200;
        javax.swing.Timer timer = new javax.swing.Timer(frameDelayMs, null);
        Runnable stopPlaying = () -> {
            timer.stop();
            playBtn.setText("▶ Play");
        };
        timer.addActionListener(e -> {
            int cur = slider.getValue();
            if (cur >= lastIdx) {
                stopPlaying.run();
            } else {
                slider.setValue(cur + 1);
            }
        });

        // Slider drives the render — one path for both manual scrub AND
        // timer ticks (the timer just calls setValue). Rebuilding the
        // Figure is 50-100 µs for typical N; well under the 200 ms budget.
        slider.addChangeListener(e -> {
            int idx = slider.getValue();
            SpectraSnapshot s = spectraHistory.get(idx);
            plotPanel.setFigure(buildSpectraFigure(s, visible));
            timeLabel.setText(String.format("t = %.4f", s.ttime()));
        });

        playBtn.addActionListener(e -> {
            if (timer.isRunning()) {
                stopPlaying.run();
            } else {
                // If we're already at the end, wrap to the start so Play
                // is always meaningful rather than a no-op.
                if (slider.getValue() >= lastIdx) slider.setValue(0);
                playBtn.setText("⏸ Pause");
                timer.start();
            }
        });

        javax.swing.JPanel controls = new javax.swing.JPanel(new java.awt.BorderLayout(8, 0));
        controls.setBorder(javax.swing.BorderFactory.createEmptyBorder(6, 8, 6, 8));
        controls.add(playBtn, java.awt.BorderLayout.WEST);
        controls.add(slider, java.awt.BorderLayout.CENTER);
        controls.add(timeLabel, java.awt.BorderLayout.EAST);

        container.add(plotPanel, java.awt.BorderLayout.CENTER);
        container.add(controls, java.awt.BorderLayout.SOUTH);
        // The refresh callback re-renders the currently-selected
        // snapshot with the current visibility array — invoked by menu
        // toggles from the frame-level View menu.
        Runnable refresh = () -> {
            int idx = slider.getValue();
            plotPanel.setFigure(buildSpectraFigure(spectraHistory.get(idx), visible));
        };
        return new SpectraTabView(container, refresh);
    }
}
