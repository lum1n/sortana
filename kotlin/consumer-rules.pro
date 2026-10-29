# JNA and the UniFFI bindings are reached through reflection.
-keep class com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
-keep class io.github.lum1n.sortana.** { *; }
