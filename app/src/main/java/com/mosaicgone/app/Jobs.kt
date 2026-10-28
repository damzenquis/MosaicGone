package com.mosaicgone.app

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.mosaicgone.core.Net
import com.mosaicgone.core.Options
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.FileInputStream
import java.nio.channels.FileChannel

data class JobRequest(val media: Uri, val isVideo: Boolean, val displayName: String, val opt: Options)

sealed interface JobStatus {
    data object Idle : JobStatus
    data class Running(val step: String, val done: Int, val total: Int, val etaSec: Long, val log: List<String>) : JobStatus
    data class Done(
        val result: Uri, val isVideo: Boolean, val source: Uri, val savedAs: String,
        val seconds: Long, val log: List<String>, val logFile: LogFile?,
        /** saved, but not everything went right (e.g. frames left uncleaned); shown under the result */
        val warning: String? = null,
    ) : JobStatus
    data class Failed(val message: String, val log: List<String>, val logFile: LogFile?) : JobStatus
    data class Cancelled(val log: List<String>, val logFile: LogFile?) : JobStatus
}

/** The job log saved as text in Download/MosaicGone. */
data class LogFile(val uri: Uri, val name: String)

/** The one job that can run at a time; shared by the service and the screens. */
object Jobs {
    val status = MutableStateFlow<JobStatus>(JobStatus.Idle)
    val preview = MutableStateFlow<Bitmap?>(null)
    @Volatile var request: JobRequest? = null
    @Volatile var cancelRequested = false

    val isBusy get() = status.value is JobStatus.Running
}

class ModelInfo(val name: String, val title: String, val description: String)

object ModelRegistry {
    /**
     * Models shown in Step 2: the restorers (ours, BasicVSR++ scaled to 32 channels / 5 blocks, and Lada's full-size
     * one), which always run with the YOLOX-Nano detector (mosaicgone_detector); their 4-frame "_t4" exports are used
     * for photos. Also
     * used internally: the Real-ESRGAN models for "Sharpen cleaned area" (realesr-animevideov3 by default;
     * realesr-general-x4v3 stored as three files with the denoise strength baked in, see Options.srFile).
     */
    val models = listOf(
        ModelInfo(Options.RESTORER, RESTORER_TITLE, "Our trained restorer. Lightweight and fast, tailored for mobile devices."),
        ModelInfo(
            Options.LADA, LADA_TITLE,
            "The original Lada restorer model by ladaapp (https://github.com/ladaapp). About 2x slower than our model on " +
                "mobile devices (depends on device's hardware).",
        ),
    )

    const val RESTORER_TITLE = "BasicVSR++, Scaled (32 channels, 5 blocks)"
    const val LADA_TITLE = "Lada BasicVSR++ generic v1.2 (64 channels, 15 blocks)"
    const val DETECTOR_TITLE = "YOLOX-Nano"

    fun find(name: String) = models.firstOrNull { it.name == name }

    /** Memory-maps the uncompressed asset straight from the APK; no copy, nothing on the Java heap. */
    fun open(ctx: Context, name: String, threads: Int): Net {
        ctx.assets.openFd("models/$name.onnx").use { afd ->
            FileInputStream(afd.fileDescriptor).channel.use { ch ->
                val buf = ch.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
                return Net.fromBuffer(buf, threads)
            }
        }
    }
}
