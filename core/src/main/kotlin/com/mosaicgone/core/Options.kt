package com.mosaicgone.core

import java.util.Locale

class OptionError(message: String) : Exception(message)

/**
 * Options for mosaic removal with MosaicGone's models (YOLOX-Nano detector, scaled BasicVSR++ restorer), plus the
 * Real-ESRGAN options that apply to sharpening the cleaned area. Field names follow the command-line flags so the
 * "More options" text box accepts the same `--flag value` syntax.
 */
data class Options(
    /** auto | clean */
    var mode: String = "auto",
    /** restorer model name without extension: [RESTORER] or [LADA] (both run with the [DETECTOR]) */
    var modelName: String = "",
    var fps: Int = 0,
    var startTime: Double = 0.0,
    var lastTime: Double = 0.0,
    /** 0~255 on the detector's probability: 128 = p > 0.5 (half the false alarms of 64 for the same finds) */
    var maskThreshold: Int = 128,
    var noFeather: Boolean = false,
    var allMosaicArea: Boolean = false,
    var medfiltNum: Int = 11,
    /**
     * Smallest mosaic that gets cleaned: width of the square around it, in processed pixels. 0 = the restorer's
     * default (32 px, its smallest training crop). Smaller ones pass through untouched; the log lists their sizes.
     */
    var minMosaicSize: Int = 0,
    var exMult: String = "auto",
    /** Real-ESRGAN -dn / --denoise_strength for --sr_model realesr-general-x4v3: 0 keeps noise, 1 strong denoise. */
    var denoiseStrength: Double = 0.5,
    /**
     * Run Real-ESRGAN on the repaired patch before it is pasted back (sharper cleaned area). Off by default.
     * Real-ESRGAN is only ever applied to the cleaned area, never to the whole picture or video.
     */
    var enhancePatch: Boolean = false,
    /** Real-ESRGAN model used by --enhance_patch: the anime-video model is the default (smooth result, 2x faster). */
    var srModel: String = SR_ANIME,
    /** Android only: short side cap for video processing; 0 = the source's size (the app only shrinks when the phone lacks memory) */
    var maxVideoSize: Int = 0,
    /** Android only: AI (ONNX Runtime) threads, 0 = auto (one per performance core). */
    var threads: Int = 0,
    /** Android only: pin AI threads to performance cores and image-handling threads to efficiency cores. */
    var affinity: Boolean = true,
    /** Run decoding/pre-processing and post-processing/encoding on their own threads, overlapping the AI. */
    var parallel: Boolean = true,
    /**
     * Android only (on by default, used only with --enhance_patch): Real-ESRGAN runs on a pool of spare efficiency-core
     * threads alongside the restorer, with the AI thread helping when the pool falls behind. Same result, less waiting.
     */
    var parallelSharpen: Boolean = true,
    /** Android only (off by default): split YUV <-> RGB conversion of each frame over several efficiency-core threads. */
    var parallelColor: Boolean = false,
) {
    /** Resolved values, filled by [resolve]. */
    var exMultValue: Double = 1.5
        private set

    /** Checks the options for the clean mode. */
    fun resolve(): Options {
        val name = modelName
        if (mode == "auto") {
            mode = if (name in RESTORERS) "clean" else throw OptionError("Please choose a model.")
        }
        if (mode !in MODES) throw OptionError("Unknown mode: $mode (auto | ${MODES.joinToString(" | ")})")
        if (name.isEmpty()) throw OptionError("Please choose a model.")
        if (name !in RESTORERS) throw OptionError("Unknown model \"$name\" (this app has: ${RESTORERS.joinToString()}).")
        exMultValue = if (exMult == "auto") 1.5
        else exMult.toDoubleOrNull() ?: throw OptionError("--ex_mult must be a number or auto")
        if (medfiltNum <= 0 || medfiltNum % 2 == 0) throw OptionError("--medfilt_num must be an odd number")
        if (maskThreshold !in 0..255) throw OptionError("--mask_threshold must be 0~255")
        if (minMosaicSize < 0) throw OptionError("--min_mosaic_size must be 0 (default) or a size in pixels")
        if (threads !in 0..64) throw OptionError("--threads must be 0 (auto) to 64")
        if (denoiseStrength < 0.0 || denoiseStrength > 1.0) throw OptionError("--denoise_strength must be between 0 and 1")
        if (srModel !in SR_MODELS) throw OptionError("--sr_model must be one of ${SR_MODELS.joinToString(" | ")}")
        return this
    }

    /** The denoise strength actually used: the app bundles the general model at 0, 0.5 and 1. */
    val denoiseLevel: Double get() = DENOISE_LEVELS.minByOrNull { kotlin.math.abs(it - denoiseStrength) }!!

    /** ONNX file (without .onnx) for a Real-ESRGAN model, with the denoise level baked in for the general model. */
    fun srFile(model: String): String =
        if (model == SR_GENERAL) "${model}_dn%03d".format(Locale.US, (denoiseLevel * 100).toInt()) else model

    /** Command line shown in the UI. */
    fun toCommandLine(mediaName: String): String = buildString {
        append("--mode $mode --media_path $mediaName --model_path ${modelName.ifEmpty { "none" }} --fps $fps")
        val d = Options()
        if (startTime != d.startTime) append(" -ss ${fmtTime(startTime)}")
        if (lastTime != d.lastTime) append(" -t ${fmtTime(lastTime)}")
        if (maskThreshold != d.maskThreshold) append(" --mask_threshold $maskThreshold")
        if (noFeather) append(" --no_feather")
        if (allMosaicArea) append(" --all_mosaic_area")
        if (medfiltNum != d.medfiltNum) append(" --medfilt_num $medfiltNum")
        if (minMosaicSize != d.minMosaicSize) append(" --min_mosaic_size $minMosaicSize")
        if (exMult != d.exMult) append(" --ex_mult $exMult")
        if (denoiseStrength != d.denoiseStrength) append(" --denoise_strength ${fmtNum(denoiseStrength)}")
        if (enhancePatch) append(" --enhance_patch")
        if (srModel != d.srModel) append(" --sr_model $srModel")
        if (maxVideoSize != d.maxVideoSize) append(" --max_video_size $maxVideoSize")
        if (threads != d.threads) append(" --threads $threads")
        if (!affinity) append(" --no_affinity")
        if (!parallel) append(" --no_parallel")
        if (!parallelSharpen) append(" --no_parallel_sharpen")
        if (parallelColor) append(" --parallel_color")
    }

    companion object {
        val MODES = listOf("clean")
        /** The restorer, BasicVSR++ scaled to 32 channels / 5 blocks (model file mosaicgone_restorer.onnx) */
        const val RESTORER = "mosaicgone_restorer"
        /** Lada's restorer (generic v1.2: BasicVSR++ 64 channels / 15 blocks, AGPL-3.0), same input and output as [RESTORER] */
        const val LADA = "lada_restorer"
        /** the restorers; each has a 16-frame file and a 4-frame "_t4" file (photos, short stretches) */
        val RESTORERS = listOf(RESTORER, LADA)
        /** The detector, YOLOX-Nano (model file mosaicgone_detector.onnx) */
        const val DETECTOR = "mosaicgone_detector"
        const val SR_GENERAL = "realesr-general-x4v3"
        const val SR_ANIME = "realesr-animevideov3"
        val SR_MODELS = listOf(SR_GENERAL, SR_ANIME)
        val DENOISE_LEVELS = listOf(0.0, 0.5, 1.0)

        private val FLAGS = setOf("no_feather", "all_mosaic_area", "no_affinity", "no_parallel", "enhance_patch", "parallel_sharpen",
            "no_parallel_sharpen", "parallel_color")

        private fun fmtNum(v: Double) = if (v == v.toLong().toDouble()) v.toLong().toString() else v.toString()

        /** Applies `--flag value` text on top of [opt]. Unknown flags are an error. */
        fun applyArgs(opt: Options, text: String) {
            val tokens = text.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
            var i = 0
            while (i < tokens.size) {
                val raw = tokens[i]
                val key = when (raw) {
                    "-ss" -> "start_time"
                    "-t" -> "last_time"
                    "-dn" -> "denoise_strength"
                    else -> if (raw.startsWith("--")) raw.substring(2) else throw OptionError("Unexpected \"$raw\" in options")
                }
                if (key in FLAGS) {
                    when (key) {
                        "no_feather" -> opt.noFeather = true
                        "all_mosaic_area" -> opt.allMosaicArea = true
                        "no_affinity" -> opt.affinity = false
                        "no_parallel" -> opt.parallel = false
                        "enhance_patch" -> opt.enhancePatch = true
                        "parallel_sharpen" -> opt.parallelSharpen = true
                        "no_parallel_sharpen" -> opt.parallelSharpen = false
                        "parallel_color" -> opt.parallelColor = true
                    }
                    i++
                    continue
                }
                val v = tokens.getOrNull(i + 1) ?: throw OptionError("Missing value for $raw")
                fun int() = v.toIntOrNull() ?: throw OptionError("$raw needs a whole number, got \"$v\"")
                fun num() = v.toDoubleOrNull() ?: throw OptionError("$raw needs a number, got \"$v\"")
                when (key) {
                    "mode" -> opt.mode = v
                    "fps" -> opt.fps = int()
                    "start_time" -> opt.startTime = parseTime(v)
                    "last_time" -> opt.lastTime = parseTime(v)
                    "mask_threshold" -> opt.maskThreshold = int()
                    "medfilt_num" -> opt.medfiltNum = int()
                    "min_mosaic_size" -> opt.minMosaicSize = int()
                    "ex_mult" -> opt.exMult = v
                    "denoise_strength" -> opt.denoiseStrength = num()
                    "sr_model" -> opt.srModel = v
                    "max_video_size" -> opt.maxVideoSize = int()
                    "threads" -> opt.threads = int()
                    "media_path", "model_path" -> {} // set by the app
                    "outscale", "tile" -> throw OptionError("$raw is not used: Real-ESRGAN only sharpens the cleaned area (--enhance_patch)")
                    else -> throw OptionError("Unknown option $raw")
                }
                i += 2
            }
        }

        /** "HH:MM:SS(.s)", "MM:SS" or plain seconds */
        fun parseTime(v: String): Double {
            val parts = v.split(":")
            var sec = 0.0
            for (p in parts) sec = sec * 60 + (p.toDoubleOrNull() ?: throw OptionError("Bad time \"$v\""))
            if (sec < 0) throw OptionError("Bad time \"$v\"")
            return sec
        }

        fun fmtTime(sec: Double): String {
            val s = sec.toLong()
            return "%02d:%02d:%02d".format(s / 3600, s / 60 % 60, s % 60)
        }
    }
}
