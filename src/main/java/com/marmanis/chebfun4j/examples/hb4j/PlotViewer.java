package com.marmanis.chebfun4j.examples.hb4j;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Standalone entry point that re-opens the NavierStokes3D analysis window
 * from a completed run's data files. Runs no simulation; just loads the
 * two named files from disk and hands them to {@link FlowStatistics#showPlots}.
 *
 * <p>Usage:
 * <pre>
 *   # Default filenames (ShortStats.dat, Spectra.dat) in the current dir
 *   mvn -P!tornado exec:java \
 *       -Dexec.mainClass=com.marmanis.chebfun4j.examples.hb4j.PlotViewer
 *
 *   # Compare a specific run's artifacts
 *   mvn -P!tornado exec:java \
 *       -Dexec.mainClass=com.marmanis.chebfun4j.examples.hb4j.PlotViewer \
 *       -Dexec.args="runs/2026-07-19/ShortStats.dat runs/2026-07-19/Spectra.dat"
 * </pre>
 *
 * <p>Positional args:
 * <ol>
 *   <li>path to a ShortStats.dat-format file (default {@code ShortStats.dat})</li>
 *   <li>path to a Spectra.dat-format file (default {@code Spectra.dat})</li>
 * </ol>
 */
public class PlotViewer {
    private static final Logger logger = LoggerFactory.getLogger(PlotViewer.class);

    public static void main(String[] args) {
        String shortStats = args.length > 0 ? args[0] : FlowStatistics.SHORT_STATS_FILE;
        String spectra    = args.length > 1 ? args[1] : FlowStatistics.SPECTRA_FILE;
        logger.info("PlotViewer — short stats: {}, spectra: {}", shortStats, spectra);
        FlowStatistics.showPlots(shortStats, spectra);
    }
}
