package com.mosaicgone.app

import ai.onnxruntime.OrtEnvironment
import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import com.mosaicgone.core.Enhancer
import com.mosaicgone.core.FrameSink
import com.mosaicgone.core.FrameSource
import com.mosaicgone.core.Gif
import com.mosaicgone.core.Img
import com.mosaicgone.core.AutoTune
import com.mosaicgone.core.MaskNet
import com.mosaicgone.core.Net
import com.mosaicgone.core.Options
import com.mosaicgone.core.PackedImages
import com.mosaicgone.core.Pipeline
import com.mosaicgone.core.Progress
import com.mosaicgone.core.Restorer
import com.mosaicgone.core.RowPool
import com.mosaicgone.core.Timing
import com.mosaicgone.core.CancelledException
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.min

/**
 * Progress -> Jobs.status / Jobs.preview, with ETA like util.counttime.
 * Keeps the full time-stamped log (saved as a .txt at the end). Called from several threads.
 */
class JobProgress(
    private val ctx: Context,
    private val timing: Timing,
    private val onChange: (JobStatus.Running) -> Unit,
) : Progress {
    private val t0 = SystemClock.elapsedRealtime()
    private val lines = ArrayList<String>()
    private var step = "Starting"
    private var stepStart = t0
    private var lastPreview = 0L
    private var lastStatus = 0L
    private var done = 0
    private var total = 0
    private var cachedStatus = ""
    private var cachedStatusAt = 0L

    @Volatile var binder: CoreBinder? = null

    val log: List<String> get() = synchronized(lines) { lines.toList() }

    /** Whole log for the .txt file */
    fun text(): String = synchronized(lines) { lines.joinToString("\n", postfix = "\n") }

    override fun step(text: String) {
        synchronized(this) {
            step = text
            stepStart = SystemClock.elapsedRealtime()
            done = 0; total = 0
        }
        log(text)
    }

    override fun progress(done: Int, total: Int) {
        val now = SystemClock.elapsedRealtime()
        val publishNow = synchronized(this) {
            this.done = done
            this.total = total
            now - lastStatus > 250 || done >= total
        }
        if (publishNow) publish()
    }

    override fun preview(img: Img) {
        val now = SystemClock.elapsedRealtime()
        synchronized(this) {
            if (now - lastPreview < 500) return
            lastPreview = now
        }
        Jobs.preview.value = timing.measure("preview") { Media.previewBitmap(img) }
    }

    override fun log(msg: String) {
        val sec = (SystemClock.elapsedRealtime() - t0) / 1000.0
        val stamp = "[${Timing.fmtDuration(sec)}]"
        synchronized(lines) {
            msg.trimEnd().lines().forEachIndexed { i, l -> lines.add(if (i == 0) "$stamp $l" else " ".repeat(stamp.length + 1) + l) }
        }
        publish()
    }

    override fun status(): String = synchronized(this) {
        val now = SystemClock.elapsedRealtime()
        if (now - cachedStatusAt > 2000) {
            cachedStatus = runCatching { DeviceStatus.describe(ctx) }.getOrDefault("")
            cachedStatusAt = now
        }
        cachedStatus
    }

    override fun heartbeat() {
        binder?.heartbeat()
    }

    override val isCancelled get() = Jobs.cancelRequested

    private fun publish() {
        val st = synchronized(this) {
            lastStatus = SystemClock.elapsedRealtime()
            val elapsed = (lastStatus - stepStart) / 1000.0
            val eta = if (done > 0 && total > done) (elapsed / done * (total - done)).toLong() else -1L
            JobStatus.Running(step, done, total, eta, emptyList())
        }
        onChange(st.copy(log = log.takeLast(40)))
    }
}

/** Runs one job on the calling (worker) thread. */
class JobRunner(
    private val ctx: Context,
    private val req: JobRequest,
    private val progress: JobProgress,
    private val timing: Timing,
) {
    private val opt = req.opt
    private val nets = ArrayList<Net>()
    private val topo = CpuTopology.current
    private val binder = CoreBinder(opt.affinity, progress::log).also { progress.binder = it }
    private var framesOut = 0

    /** set when the result was saved but some of it could not be cleaned (see [noteUncleaned]) */
    var warning: String? = null
        private set

    /** Frames that had a mosaic but were saved unchanged (the log says why): the result screen says so too. */
    private fun noteUncleaned(pipe: Pipeline, total: Int) {
        val n = pipe.uncleanedFrames
        if (n > 0) warning = "$n of $total frames could not be cleaned and were saved unchanged (details in the log)."
    }

    /** AI threads: --threads, else one per performance core (or up to 4 on phones with equal cores). */
    val threads: Int = when {
        opt.threads > 0 -> opt.threads
        topo.heterogeneous -> topo.perf.size
        else -> min(4, topo.cores.size)
    }.coerceIn(1, maxOf(1, topo.cores.size))

    private fun baseName(): String = req.displayName.substringBeforeLast('.').ifEmpty { "result" }

    private fun suffix(): String = "clean"

    val logName: String get() = "${baseName()}_${suffix()}_log.txt"

    private fun header() {
        val pi = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        val am = ctx.getSystemService(ActivityManager::class.java)
        val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val soc = if (Build.VERSION.SDK_INT >= 31) " · SoC ${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else ""
        val ortVersion = runCatching { OrtEnvironment.getEnvironment().version }.getOrDefault("?")
        progress.log(
            """
            MosaicGone v${pi.versionName} (build ${pi.longVersionCode}) · job log
            Started ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())}
            Device: ${Build.MANUFACTURER} ${Build.MODEL}$soc · Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})
            Memory: ${"%.1f".format(Locale.US, mem.totalMem / 1e9)} GB RAM, ${"%.1f".format(Locale.US, mem.availMem / 1e9)} GB free, app heap limit ${Runtime.getRuntime().maxMemory() / (1 shl 20)} MB
            CPU: ${topo.describe()}
            AI engine: ONNX Runtime $ortVersion (CPU) · threads $threads ${if (opt.threads > 0) "(--threads)" else "(auto)"} · idle spinning off · image handling ${if (opt.parallel) "in parallel threads" else "sequential (--no_parallel)"}
            Affinity: ${binder.status}
            Device state: ${progress.status()}
            Media: ${req.displayName} (${if (req.isVideo) "video" else "image"})
            Options: ${opt.toCommandLine(req.displayName)}
            """.trimIndent(),
        )
        if (binder.active && threads > topo.perf.size) {
            progress.log("   note: $threads AI threads on ${topo.perf.size} performance cores; threads will share cores")
        }
    }

    private fun load(name: String, threads: Int = this.threads): Net {
        progress.log("Loading model $name" + if (threads != this.threads) " ($threads thread per call)" else "")
        val before = Affinity.threadIds()
        val net = timing.measure("load") { ModelRegistry.open(ctx, name, threads) }
        binder.adoptOrtThreads(before)
        return net.also { nets.add(it) }
    }

    /**
     * Efficiency cores left for helper pools (--parallel_sharpen, --parallel_color) besides the reader and writer
     * threads; on phones without a big/little split, the cores the AI threads do not use.
     */
    private val spareCores: Int = if (topo.heterogeneous) topo.eff.size - 2 else topo.cores.size - threads - 2

    /** --parallel_sharpen: Real-ESRGAN pool size (0 = sharpening on the AI thread only, e.g. no spare cores) */
    private val sharpenThreads: Int = if (opt.parallelSharpen && opt.enhancePatch && spareCores >= 1) spareCores.coerceIn(1, 6) else 0

    /** --parallel_color: helper threads for YUV <-> RGB, shared by the reader and writer */
    private val colorPool: RowPool? = if (opt.parallelColor && spareCores >= 1) {
        RowPool(spareCores.coerceIn(1, 4)) { binder.bindCurrentAsIo() }
    } else null

    /** frame rate of the processed frames, for the detection timeline in the log */
    private var frameRate = 0.0

    /** --enhance_patch (off by default): Real-ESRGAN only on the cleaned area, never on the whole picture/video. */
    private fun patchEnhancer(): Enhancer? {
        if (!opt.enhancePatch) return null
        if (opt.srModel == Options.SR_GENERAL && opt.denoiseLevel != opt.denoiseStrength) {
            progress.log("   denoise strength ${opt.denoiseStrength} -> ${opt.denoiseLevel} (bundled levels: 0, 0.5, 1)")
        }
        progress.log(
            "   sharpening the cleaned area with Real-ESRGAN ${opt.srFile(opt.srModel)}" +
                if (sharpenThreads > 0) " on $sharpenThreads parallel thread(s)${if (binder.active) " (efficiency cores)" else ""} " +
                    "with the AI thread (performance cores) helping (--parallel_sharpen)" else " on the AI thread (performance cores)",
        )
        // in the pool each call runs on its own pool thread (1 ONNX Runtime thread), so it never competes for the AI cores
        return Enhancer(if (sharpenThreads > 0) load(opt.srFile(opt.srModel), threads = 1) else load(opt.srFile(opt.srModel)))
    }

    /** YOLOX-Nano, which finds the mosaic */
    private fun maskNet(): MaskNet = MaskNet(load(Options.DETECTOR))

    /** The chosen restorer (ours or Lada's); [video]: also the 16-frame export (photos only need the 4-frame one) */
    private fun restorer(video: Boolean) =
        Restorer(if (video) load(opt.modelName) else null, lazy(LazyThreadSafetyMode.NONE) { load(opt.modelName + "_t4") })

    private fun pipeline() = Pipeline(
        opt, progress, timing = timing, threadInit = { binder.bindCurrentAsIo() }, patchSr = patchEnhancer(),
        sharpenThreads = sharpenThreads, frameRate = frameRate,
    )

    fun run(): Pair<Uri, String> {
        val wall0 = System.nanoTime()
        binder.bindCurrentAsAi()
        header()
        colorPool?.let { progress.log("   colour conversion split over ${it.threads + 1} threads (--parallel_color)") }
        if (opt.parallelColor && colorPool == null) progress.log("   --parallel_color: no spare cores on this phone, colour conversion stays on one thread")
        try {
            return when {
                req.isVideo -> runVideo()
                Media.isGif(ctx, req.media) -> runGif()
                else -> runImage()
            }
        } finally {
            colorPool?.close()
            nets.forEach { runCatching { it.close() } }
            progress.log(Timing.report("Whole job", framesOut, System.nanoTime() - wall0, timing.snapshot()))
            progress.log(binder.summary())
            progress.log("Device state at end: ${progress.status()}")
        }
    }

    /** Java heap still free for pictures, keeping some for the models' input/output buffers and the app itself. */
    private fun heapBudget(): Long {
        val rt = Runtime.getRuntime()
        val free = rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())
        return maxOf(32L shl 20, free - (96L shl 20))
    }

    private fun mb(bytes: Long) = bytes shr 20

    private fun runImage(): Pair<Uri, String> {
        progress.step("Reading image")
        // no size limit of our own: full size unless this phone's app memory cannot hold it
        // (pixels live in the Java heap ~5 bytes each: the RGB image, the mask, small buffers; bitmaps are outside it)
        val maxPixels = (heapBudget() / 5).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        val photo = timing.measure("read") { Media.decodePhoto(ctx, req.media, maxPixels) }
        val img = timing.measure("read") { Media.toImg(photo.bitmap) }
        photo.bitmap.recycle()
        progress.log(
            String.format(
                Locale.US, "Image %dx%d %s, %.2f bits/pixel · colour space %s%s", photo.sourceWidth, photo.sourceHeight,
                photo.mime.removePrefix("image/").uppercase(), photo.sourceBpp, photo.colorSpace.name,
                if (img.w != photo.sourceWidth) " · processing at ${img.w}x${img.h}: this phone gives the app ${mb(Runtime.getRuntime().maxMemory())} MB, " +
                    "not enough for ${photo.sourceWidth.toLong() * photo.sourceHeight / 1_000_000} MP at full size" else "",
            ),
        )
        val pipe = pipeline()
        val netM = maskNet()
        val out = pipe.cleanMosaicImg(img, netM, restorer(video = false), inPlace = true)
        progress.preview(out)
        progress.step("Saving")
        val saved = timing.measure("save") { Media.savePhoto(ctx, out, "${baseName()}_${suffix()}", photo) }
        progress.log("   saved as ${saved.note}")
        framesOut = 1
        return saved.uri to "Pictures/MosaicGone/${saved.name}"
    }

    /**
     * Animated GIF: every frame is cleaned like a video and written back as GIF with the source's size, frame delays,
     * loop count and per-frame palettes.
     */
    private fun runGif(): Pair<Uri, String> {
        progress.step("Reading GIF")
        val bytes = Media.readBytes(ctx, req.media)
        val anim = timing.measure("read") { Gif.decode(bytes) }
        val totalCs = anim.frames.sumOf { it.delayCs }
        progress.log(
            String.format(
                Locale.US, "GIF %dx%d, %d frame(s), %.1f s, %s, %d bytes%s", anim.width, anim.height, anim.frames.size, totalCs / 100.0,
                if (anim.loopCount == 0) "loops forever" else if (anim.loopCount > 0) "loops ${anim.loopCount}x" else "plays once", bytes.size,
                if (anim.hasTransparency) " · has transparency" else "",
            ),
        )
        if (totalCs > 0) frameRate = anim.frames.size * 100.0 / totalCs
        // frames stay deflated in memory (source and result), so long GIFs fit
        val frames = anim.frames
        val open = {
            object : FrameSource {
                var i = 0
                override val estimatedFrames = frames.size
                override fun next() = if (i < frames.size) frames[i++].image else null
                override fun close() {}
            }
        }
        val out = PackedImages()
        val sink = object : FrameSink {
            override fun write(img: Img) { out.add(img) }
            override fun close() {}
        }
        val pipe = pipeline()
        val netM = maskNet()
        pipe.cleanMosaicVideoRestorer(open, sink, restorer(video = true), netM)
        noteUncleaned(pipe, out.size)
        progress.step("Saving")
        val gif = timing.measure("save") { Gif.encode(anim, out) }
        val name = "${baseName()}_${suffix()}.gif"
        val uri = timing.measure("save") { Media.saveGif(ctx, gif, name) }
        progress.log(String.format(Locale.US, "   saved GIF with the source palettes: %d bytes (source %d, %.0f%%)", gif.size, bytes.size, 100.0 * gif.size / bytes.size))
        framesOut = out.size
        return uri to "Pictures/MosaicGone/$name"
    }

    private fun runVideo(): Pair<Uri, String> {
        progress.step("Step:1/4 -- Reading video")
        val info = VideoProbe.probe(ctx, req.media)
        var (outW, outH) = info.outputSize(opt.maxVideoSize)
        // no size limit of our own: the source's size unless this phone's app memory cannot hold the frames in flight
        // (AutoTune.HELD_FRAMES; ~16 more bytes/pixel of work buffers)
        val need = outW.toLong() * outH * (3 * AutoTune.HELD_FRAMES + 16)
        val budget = heapBudget()
        var memoryNote = ""
        if (need > budget) {
            val s = kotlin.math.sqrt(budget.toDouble() / need)
            val fit = info.outputSize(maxOf(2, (minOf(outW, outH) * s).toInt()))
            memoryNote = " (reduced from ${outW}x$outH: this phone gives the app ${mb(Runtime.getRuntime().maxMemory())} MB, the source size needs ~${mb(need)} MB)"
            outW = fit.first; outH = fit.second
        }
        val outFps = if (opt.fps > 0) min(opt.fps.toDouble(), info.fps) else info.fps
        val startUs = (opt.startTime * 1e6).toLong()
        val endUs = if (opt.lastTime > 0) startUs + (opt.lastTime * 1e6).toLong() else 0L
        if (startUs >= info.durationUs && info.durationUs > 0) throw IllegalArgumentException("Start time is after the end of the video.")
        progress.log(
            "Video ${info.displayWidth}x${info.displayHeight} @ %.2f fps, %.1f s, %s%s -> processing at ${outW}x$outH$memoryNote, %.2f fps"
                .format(
                    Locale.US, info.fps, info.durationUs / 1e6, info.mime.removePrefix("video/").uppercase(),
                    if (info.color.spec.isHdr) " HDR ${info.color.spec.transfer}" else if (info.tenBit) " 10-bit" else "", outFps,
                ),
        )
        frameRate = outFps
        val pts = ArrayList<Long>()
        var decoderLogged = false
        val open = {
            val announce = !decoderLogged
            decoderLogged = true
            VideoFrameSource(ctx, req.media, info, outW, outH, startUs, endUs, opt.fps, pts, timing, progress::log, colorPool, announce)
        }
        val tmp = File(ctx.cacheDir, "mosaicgone_out.mp4").apply { delete() }
        val sink = VideoFrameSink(ctx, tmp, outFps, pts, if (info.hasAudio) req.media else null, startUs, endUs, progress::log, timing, info, colorPool)
        try {
            try {
                sink.use {
                    val pipe = pipeline()
                    val netM = maskNet()
                    pipe.cleanMosaicVideoRestorer(open, sink, restorer(video = true), netM)
                    noteUncleaned(pipe, sink.frameCount)
                    progress.step("Step:4/4 -- Convert images to video")
                    if (sink.frameCount == 0) throw IllegalStateException("No frames were produced.")
                }
            } catch (e: Throwable) {
                // the sink has closed what it wrote into a playable file: keep it instead of losing hours of work
                framesOut = sink.writtenCount
                if (e !is CancelledException && sink.outputs.isNotEmpty() && sink.writtenCount > 0) {
                    val saved = runCatching { saveVideos(sink.outputs, "${baseName()}_${suffix()}_partial") }.getOrNull()
                    if (saved != null) {
                        progress.log(String.format(Locale.US, "Saved the part finished before the error (%d frames, %.1f s) to %s", sink.writtenCount, sink.writtenCount / outFps, saved.second))
                        throw IOException("${e.message ?: e}\n\nThe part finished before the error was saved to ${saved.second}.", e)
                    }
                }
                throw e
            }
            framesOut = sink.writtenCount
            progress.step("Saving")
            return saveVideos(sink.outputs, "${baseName()}_${suffix()}")
        } finally {
            tmp.delete()
            sink.outputs.forEach { it.delete() }
        }
    }

    /** Saves the result file(s) to Movies/MosaicGone; returns the first one. */
    private fun saveVideos(files: List<File>, base: String): Pair<Uri, String> {
        var first: Pair<Uri, String>? = null
        for ((i, f) in files.withIndex()) {
            val name = if (i == 0) "$base.mp4" else "${base}_part${i + 1}.mp4"
            val uri = timing.measure("save") { Media.saveVideo(ctx, f, name) }
            if (first == null) first = uri to "Movies/MosaicGone/$name"
            if (i > 0) progress.log("   also saved Movies/MosaicGone/$name")
        }
        return first ?: throw IllegalStateException("No frames were produced.")
    }
}
