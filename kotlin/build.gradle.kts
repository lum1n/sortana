plugins {
    // AGP 9 compiles Kotlin itself; no separate Kotlin plugin.
    id("com.android.library") version "9.4.1"
}

android {
    namespace = "io.github.lum1n.sortana"
    compileSdk = 36
    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
    }
}

dependencies {
    // The bindings load libsortana_ffi.so through JNA; suspend functions need coroutines.
    implementation("net.java.dev.jna:jna:5.17.0@aar")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
}
