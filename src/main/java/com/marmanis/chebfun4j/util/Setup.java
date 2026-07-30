package com.marmanis.chebfun4j.util;

public class Setup {
    public static int envIntOr(String name, int def) {
        String v = System.getenv(name);
        try { return v == null ? def : Integer.parseInt(v.trim()); }
        catch (NumberFormatException e) { return def; }
    }

    public static Integer envIntOrNull(String name) {
        String v = System.getenv(name);
        if (v == null) return null;
        try { return Integer.parseInt(v.trim()); }
        catch (NumberFormatException e) { return null; }
    }

    public static Long envLongOrNull(String name) {
        String v = System.getenv(name);
        if (v == null) return null;
        try { return Long.parseLong(v.trim()); }
        catch (NumberFormatException e) { return null; }
    }
}
