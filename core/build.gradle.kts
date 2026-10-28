plugins {
    id("org.jetbrains.kotlin.jvm")
}

kotlin {
    jvmToolchain(17)
}

val ortVersion = "1.20.0"

dependencies {
    // The app supplies onnxruntime-android at runtime; the desktop build is only for tests on the PC.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("junit:junit:4.13.2")
}

tasks.test {
    maxHeapSize = "3g"
    testLogging {
        events("passed", "failed")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
