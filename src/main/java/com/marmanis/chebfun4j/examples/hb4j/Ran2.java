package com.marmanis.chebfun4j.examples.hb4j;

/**
 * A Java implementation of the Bays-Durham shuffle pseudo-random number generator,
 * matching the behavior of the standard ran2 algorithm from Numerical Recipes in Fortran.
 * It is used to generate identical deterministic random initial conditions for Navier-Stokes simulations.
 */
public class Ran2 {
    private int idum;
    private int idum2 = 123456789;
    private int iy = 0;
    private final int[] iv = new int[32];

    private static final int IM1 = 2147483563;
    private static final int IM2 = 2147483399;
    private static final int IMM1 = IM1 - 1;
    private static final int IA1 = 40014;
    private static final int IA2 = 40692;
    private static final int IQ1 = 53668;
    private static final int IQ2 = 52774;
    private static final int IR1 = 12211;
    private static final int IR2 = 3791;
    private static final int NTAB = 32;
    private static final int NDIV = 1 + IMM1 / NTAB;
    private static final double AM = 1.0 / IM1;
    private static final double EPS = 1.2e-7;
    private static final double RNMX = 1.0 - EPS;

    /**
     * Initializes the random number generator with a negative integer seed.
     *
     * @param idum the initial seed
     */
    public Ran2(int idum) {
        this.idum = Math.max(-idum, 1);
        this.idum2 = this.idum;
        for (int j = NTAB + 7; j >= 0; j--) {
            int k = this.idum / IQ1;
            this.idum = IA1 * (this.idum - k * IQ1) - k * IR1;
            if (this.idum < 0) this.idum += IM1;
            if (j < NTAB) iv[j] = this.idum;
        }
        iy = iv[0];
    }

    /**
     * Returns a uniform pseudo-random number in the range [0.0, 1.0).
     *
     * @return the next pseudo-random value
     */
    public double next() {
        int k = idum / IQ1;
        idum = IA1 * (idum - k * IQ1) - k * IR1;
        if (idum < 0) idum += IM1;

        k = idum2 / IQ2;
        idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
        if (idum2 < 0) idum2 += IM2;

        int j = iy / NDIV;
        iy = iv[j] - idum2;
        iv[j] = idum;
        if (iy < 1) iy += IMM1;

        return Math.min(AM * iy, RNMX);
    }
}
