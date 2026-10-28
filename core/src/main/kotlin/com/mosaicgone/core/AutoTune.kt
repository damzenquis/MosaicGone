package com.mosaicgone.core

import java.util.Locale
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * "Auto-Select Options": picks the More Options values for this phone, this file and FPS.
 *
 * Costs are for the minimum device (Galaxy A17 class, 2 performance cores at 2.4 GHz, 6 efficiency cores at
 * 2.0 GHz), measured there for YOLOX-Nano and the scaled BasicVSR++ (see [RESTORER]).
 * The estimate assumes a mosaic in every frame, so it is an upper bound.
 */
object AutoTune {

    class Device(
        val perfCores: Int, val effCores: Int, val perfGHz: Double, val effGHz: Double,
        /** distinct performance and efficiency cores */
        val heterogeneous: Boolean,
        val ramGB: Double, val heapMB: Long,
    ) {
        val cores get() = if (heterogeneous) perfCores + effCores else max(perfCores, effCores)
        /** AI speed relative to the reference phone (its 2 x 2.4 GHz performance cores) */
        val aiSpeed: Double get() = (perfCores.coerceAtMost(4) * perfGHz / (2 * 2.4)).coerceIn(0.25, 6.0)
        /** threads left for helper pools besides the reader and writer (matches JobRunner.spareCores) */
        fun spare(aiThreads: Int) = if (heterogeneous) effCores - 2 else cores - aiThreads - 2
    }

    enum class Kind { IMAGE, GIF, VIDEO }

    class Media(
        val kind: Kind, val width: Int, val height: Int,
        /** source frame rate (videos / GIFs) */
        val fps: Double = 0.0,
        val durationSec: Double = 0.0,
        val hdr: Boolean = false,
        val tenBit: Boolean = false,
    ) {
        val pixels get() = width.toLong() * height
        val shortSide get() = minOf(width, height)
    }

    class Result(val options: Options, val reasons: List<String>, val estimateSec: Double?)

    /**
     * Per frame on the A17, measured (v0.1 log of 255.mp4, 848x478): the detection pass 0.2 s (with its decode and
     * mask work), the restorer 15.1 s per 16-frame call with 4 shared = 1.3 s per cleaned frame. Lada's restorer:
     * 2.25x ours (PC, 2 threads: 10.6 s vs 4.7 s per call).
     */
    private const val DETECTOR = 0.2
    private const val RESTORER = 1.3
    private const val LADA = RESTORER * 2.25
    /** Real-ESRGAN on the patch, per frame on one A17 performance core (android.5 / .7 logs): anime 2.26 s, general 3.75 s */
    private const val SR_ANIME = 2.26
    private const val SR_GENERAL = 3.75
    /**
     * An efficiency core (A55 class) runs Real-ESRGAN this much slower than a performance core per GHz. Measured on
     * the A17 (android.7 log of 255.mp4): 11.96 s per patch on an efficiency core vs 2.26 s on a performance core.
     */
    private const val LITTLE_PENALTY = 4.5

    /**
     * Extra seconds per cleaned frame that sharpening adds with --parallel_sharpen: the pool of [spare] efficiency cores
     * and the AI thread share the patches so both finish together (the AI thread also runs the restorer, [clean] s).
     */
    internal fun parallelSharpenCost(sr: Double, clean: Double, spare: Int, effGHz: Double): Double {
        if (spare < 1) return sr
        val pool = sr * LITTLE_PENALTY * (2.4 / effGHz.coerceAtLeast(0.5)) / spare
        // share f on the AI thread: clean + sr * f = pool * (1 - f)
        val f = ((pool - clean) / (sr + pool)).coerceIn(0.0, 1.0)
        return sr * f
    }

    /**
     * [current]: the form as it is (model, FPS, start/duration, the extra text box are kept);
     * returns a copy with the More Options values chosen.
     */
    fun choose(device: Device, media: Media, current: Options): Result {
        val o = current.copy()
        val why = ArrayList<String>()
        val video = media.kind != Kind.IMAGE
        val aiThreads = if (o.threads > 0) o.threads else if (device.heterogeneous) device.perfCores else minOf(4, device.cores)
        val spare = device.spare(aiThreads)

        // the defaults that scored best in testing; threshold 128 = p > 0.5 (half the false alarms of 64, bench_vs_71)
        o.maskThreshold = 128; o.noFeather = false; o.allMosaicArea = false; o.exMult = "auto"
        o.threads = 0; o.parallel = true
        o.maxVideoSize = 0
        why += "Detection, edge feathering and area expansion at the defaults that scored best in testing (threshold 128, feather on, expansion auto)."
        why += "Processed at the source's own size (Max video size 0); the app only reduces it if this phone runs short of memory."

        o.affinity = device.heterogeneous
        why += if (device.heterogeneous) "AI pinned to the ${device.perfCores} performance cores (auto threads)."
        else "All cores are equal: no core binding needed; AI threads auto."

        // smoothing window ≈ 0.44 s of processed video: 11 frames at 25 fps, shorter at low FPS so the cleaned square
        // follows fast motion sooner, longer at high FPS to stay steady
        val outFps = when {
            media.kind == Kind.IMAGE -> 0.0
            o.fps > 0 && media.fps > 0 -> minOf(o.fps.toDouble(), media.fps)
            o.fps > 0 -> o.fps.toDouble()
            else -> media.fps
        }
        if (video && outFps > 0) {
            o.medfiltNum = ((outFps * 0.44).roundToInt() or 1).coerceIn(3, 15)
            why += String.format(Locale.US, "Smoothing window %d frames (≈0.44 s at %.0f fps).", o.medfiltNum, outFps)
        } else o.medfiltNum = 11

        // QualityTest.mosaicGoneVsClassic: sharpening on top of the restorer lowered PSNR on 3 of 5 clips, more flicker, ~35% slower
        o.srModel = Options.SR_ANIME
        o.denoiseStrength = 0.5
        o.enhancePatch = false
        why += "Sharpening off: the restorer already draws sharp detail; sharpening on top scored lower in testing (more flicker) and is slower."
        val clean = (if (o.modelName == Options.LADA) LADA else RESTORER) / device.aiSpeed
        val srAlone = SR_GENERAL / device.aiSpeed
        o.parallelSharpen = spare >= 1
        if (video) why += if (o.parallelSharpen) String.format(
            Locale.US, "If you turn sharpening on, it runs in parallel on %d spare core(s) with the AI thread helping: about +%.1f s per frame (realesr-general) instead of +%.1f s.",
            spare, parallelSharpenCost(srAlone, clean, spare, device.effGHz), srAlone,
        ) else String.format(Locale.US, "If you turn sharpening on, it adds about %.1f s per frame here (no spare cores for a parallel pool).", srAlone)

        o.minMosaicSize = 0
        if (video) why += "Minimum mosaic size: the restorer's default (32 px squares, its smallest training crop), so small mosaics get cleaned too."

        // colour conversion over several threads: worth it from 720p up (or HDR, which costs more per pixel)
        o.parallelColor = video && spare >= 1 && (media.pixels >= 1280L * 720 || media.hdr)
        if (o.parallelColor) why += "Colour conversion split over spare cores (${media.width}x${media.height}${if (media.hdr) " HDR" else ""})."

        if (media.hdr) why += "HDR video: converted to standard (SDR) colours."
        else if (media.tenBit) why += "10-bit video: read at full precision."

        // memory: the same check the job makes, so the user knows up front
        if (video) {
            val needMB = media.pixels * (3 * HELD_FRAMES + 16) / (1 shl 20)
            if (needMB > device.heapMB - 96) why += "Note: this phone gives the app ${device.heapMB} MB; the video will be processed somewhat smaller than its source (needs ~$needMB MB at full size)."
        }

        val estimate = if (video && outFps > 0 && media.durationSec > 0) {
            val dur = if (o.lastTime > 0) minOf(o.lastTime, max(0.0, media.durationSec - o.startTime)) else max(0.0, media.durationSec - o.startTime)
            dur * outFps * (DETECTOR / device.aiSpeed + clean)
        } else null
        return Result(o, why, estimate)
    }

    /**
     * Full-size frames a video job holds at once (JobRunner's memory check): the restorer keeps only 256 px crops
     * waiting; its two readers each hold a frame or two, plus the stage queues.
     */
    const val HELD_FRAMES = 10

    fun fmtEstimate(sec: Double): String {
        val m = (sec / 60).roundToInt()
        return when {
            sec < 90 -> "about ${sec.roundToInt()} s"
            m < 90 -> "about $m min"
            else -> String.format(Locale.US, "about %.1f h", sec / 3600)
        }
    }
}
