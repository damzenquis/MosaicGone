import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val keystoreProps = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

// The license and third-party notices go into the APK (assets/legal/) so the app can show them (Credits row → License).
// Copied from the repository root at build time (copyLegalNotices below): the root files stay the only copies to edit.
val legalAssets = layout.buildDirectory.dir("generated/legal")

android {
    namespace = "com.mosaicgone.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.mosaicgone.app"
        minSdk = 29
        targetSdk = 35
        // MosaicGone v0.1: a new app (new applicationId), so the version count starts again
        versionCode = 5
        versionName = "0.5"
        ndk {
            // Galaxy A17 and nearly every current phone are arm64; armeabi-v7a covers older 32-bit ones.
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    signingConfigs {
        if (keystoreProps.containsKey("storeFile")) {
            create("release") {
                storeFile = rootProject.file(keystoreProps.getProperty("storeFile"))
                storePassword = keystoreProps.getProperty("storePassword")
                keyAlias = keystoreProps.getProperty("keyAlias")
                keyPassword = keystoreProps.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    androidResources {
        // keep models uncompressed so they can be memory-mapped straight from the APK
        noCompress += "onnx"
    }

    sourceSets["main"].assets.srcDir(legalAssets)
}

val copyLegalNotices by tasks.registering(Copy::class) {
    into(legalAssets.map { it.dir("legal") })
    from(rootProject.file("LICENSE")) { rename { "LICENSE.txt" } }
    from(rootProject.file("THIRD_PARTY_NOTICES.md"))
    from(rootProject.file("third_party_licenses")) { into("third_party_licenses") }
}
tasks.named("preBuild") { dependsOn(copyLegalNotices) }

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    // calls libc sched_setaffinity for pinning threads to performance cores (no NDK build needed)
    implementation("net.java.dev.jna:jna:5.14.0@aar")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    testImplementation("junit:junit:4.13.2")
}
