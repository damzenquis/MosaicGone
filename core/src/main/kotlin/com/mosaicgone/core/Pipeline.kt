package com.mosaicgone.core

import java.io.ByteArrayOutputStream
import java.util.Locale
import java.util.zip.Deflater
import java.util.zip.Inflater
import kotlin.math.max

class CancelledException : Exception("Cancelled")

/**
 * Progress reporting, replacing the console bar and the cv2 preview windows.
 * May be called from the reader, AI and writer threads, so implementations must be thread-safe.
 */
interface Progress {
    fun step(text: String) {}
    fun progress(done: Int, total: Int) {}
    fun preview(img: Img) {}
    fun log(msg: String) {}
    /** one-line device state (thermal, clocks, battery) for the log; empty if unknown */
    fun status(): String = ""
    /** called every few frames; the app re-applies CPU affinity here */
    fun heartbeat() {}
    val isCancelled: Boolean get() = false
}

/** Sequential decoded frames (RGB). [next] returns null at the end. Used from one thread at a time. */
interface FrameSource : AutoCloseable {
    /** best guess before decoding, for the progress bar */
    val estimatedFrames: Int
    fun next(): Img?
}

interface FrameSink : AutoCloseable {
    fun write(img: Img)
}

/**
 * Images kept in memory deflated (binary masks compress ~100x, GIF frames several times), so long videos and
 * GIFs do not need memory for every frame. Replaces the mask folders in ./tmp. [get] returns a fresh copy.
 */
class PackedImages : AbstractList<Img>() {
    private class Item(val w: Int, val h: Int, val c: Int, val z: ByteArray)
    private val items = ArrayList<Item>()
    override val size get() = items.size

    fun add(m: Img) = synchronized(items) { items.add(Item(m.w, m.h, m.c, pack(m.data))) }

    override fun get(index: Int): Img {
        val item = synchronized(items) { items[index] }
        return Img(item.w, item.h, item.c).also { unpack(item.z, it.data) }
    }

    companion object {
        fun pack(data: ByteArray): ByteArray {
            val d = Deflater(1)
            d.setInput(data)
            d.finish()
            val bos = ByteArrayOutputStream(maxOf(1024, data.size / 16))
            val buf = ByteArray(8192)
            while (!d.finished()) bos.write(buf, 0, d.deflate(buf))
            d.end()
            return bos.toByteArray()
        }

        fun unpack(z: ByteArray, out: ByteArray) {
            val inf = Inflater()
            inf.setInput(z)
            var off = 0
            while (off < out.size && !inf.finished()) off += inf.inflate(out, off, out.size - off)
            inf.end()
        }
    }
}

/**
 * Mosaic removal with YOLOX-Nano (finding) and the scaled BasicVSR++ (restoring), plus optional Real-ESRGAN
 * sharpening of the cleaned area. Video passes run as three overlapping stages (see [Staged]): read + pre-process |
 * AI | post-process + write. Every stage is timed into [timing]; each step ends with a breakdown table in the log.
 */
class Pipeline(
    private val opt: Options,
    private val progress: Progress,
    val timing: Timing = Timing(),
    threadInit: (String) -> Unit = {},
    /** --enhance_patch: Real-ESRGAN for the repaired patch (on the AI thread, or on [sharpenThreads] threads with --parallel_sharpen) */
    private val patchSr: Enhancer? = null,
    /** --parallel_sharpen pool size; its net should have 1 intra-op thread so each call stays on its pool thread */
    sharpenThreads: Int = 0,
    /** source frame rate after FPS thinning, for the detection timeline in the log (0 = frame numbers only) */
    private val frameRate: Double = 0.0,
) {
    private val threadInit = threadInit
    // the caller sizes the pool from the spare cores (0 = none: sharpening stays on the AI thread)
    private val sharpenThreads = if (opt.parallelSharpen && patchSr != null) maxOf(0, sharpenThreads) else 0
    private val staged = Staged(opt.parallel, threadInit, depth = 2)
    private val cancelled = { progress.isCancelled }

    private inline fun <T> t(key: String, block: () -> T): T = timing.measure(key, block)

    // ------------------------------------------------------------------ step bookkeeping

    private var stepName = ""
    private var stepStart = 0L
    private var stepSnap: Map<String, Timing.Stat> = emptyMap()

    private fun beginStep(name: String) {
        stepName = name
        stepStart = System.nanoTime()
        stepSnap = timing.snapshot()
        progress.step(name)
        progress.status().takeIf { it.isNotEmpty() }?.let { progress.log("   $it") }
    }

    private fun endStep(frames: Int) {
        progress.log(Timing.report(stepName, frames, System.nanoTime() - stepStart, Timing.diff(timing.snapshot(), stepSnap)))
    }

    /** Frame counter for the writer stage: progress bar, heartbeat, and a rate line every 10%. */
    private inner class Counter(private val total: Int) {
        var done = 0
            private set
        private val start = System.nanoTime()
        private var markTime = start
        private var markDone = 0
        private val stride = max(1, total / 10)

        fun tick() {
            done++
            progress.progress(done, max(total, done))
            if (done % 25 == 0) progress.heartbeat()
            if (done % stride == 0 && done < total) {
                val now = System.nanoTime()
                val avg = (now - start) / 1e6 / done
                val recent = (now - markTime) / 1e6 / (done - markDone)
                val st = progress.status().let { if (it.isEmpty()) "" else " · $it" }
                progress.log(String.format(Locale.US, "   %d/%d · %.0f ms/frame avg · last %d: %.0f ms/frame%s", done, total, avg, done - markDone, recent, st))
                markTime = now
                markDone = done
            }
        }
    }

    /**
     * --enhance_patch: Real-ESRGAN x4 on the restored 256 px patch first, so stretching it to the mosaic area is a
     * downscale instead of a blurry upscale.
     */
    private fun sharpen(fake: Img, size: Int): Img {
        val sr = patchSr ?: return fake
        // every cleaned frame is sharpened (no size cut-off): mixing sharpened and plain frames flickers
        return t("enhance") {
            // mirror-pad all sides first (like RealESRGANer's pre_pad) so the net does not ring at the patch border
            val p = PATCH_PAD
            val big = sr.enhance(ImageOps.reflectPad(fake, p))
                .crop(p * Enhancer.SCALE, p * Enhancer.SCALE, fake.w * Enhancer.SCALE, fake.h * Enhancer.SCALE)
            // the x4 result is usually larger than the area: shrink it by area averaging, not cubic (which aliases)
            if (big.w > size * 2) ImageOps.resizeArea(big, size * 2, size * 2) else big
        }
    }

    /**
     * --parallel_sharpen: Real-ESRGAN calls on their own threads (the spare efficiency cores); the writer picks the
     * results up in frame order. When every pool thread is busy and one more patch is already waiting, the AI thread
     * sharpens the next one itself instead of letting the backlog grow: never slower than without the pool.
     */
    private inner class SharpenPool : AutoCloseable {
        private val n = java.util.concurrent.atomic.AtomicInteger()
        private val ex = java.util.concurrent.Executors.newFixedThreadPool(sharpenThreads) { r ->
            Thread({ threadInit("sharpen"); r.run() }, "${Timing.SHARPEN_THREAD}-${n.incrementAndGet()}").apply { isDaemon = true }
        }
        private val pending = java.util.concurrent.atomic.AtomicInteger()
        /** calls running on a pool thread right now (queued ones not counted) */
        private val running = java.util.concurrent.atomic.AtomicInteger()
        var onPool = 0
            private set
        var onAiThread = 0
            private set

        /** Called from the AI thread only. */
        fun submit(fake: Img, size: Int): java.util.concurrent.Future<Img> {
            if (pending.get() > sharpenThreads) {
                onAiThread++
                return java.util.concurrent.CompletableFuture.completedFuture(sharpen(fake, size))
            }
            onPool++
            pending.incrementAndGet()
            return ex.submit(java.util.concurrent.Callable {
                running.incrementAndGet()
                try { sharpen(fake, size) } finally { running.decrementAndGet(); pending.decrementAndGet() }
            })
        }

        fun summary() = "   sharpening: $onPool frame(s) on the $sharpenThreads pool thread(s), $onAiThread on the AI thread while the pool was busy (--parallel_sharpen)"

        /**
         * Drops the queued calls and waits for the running ones, however long they take: the caller closes the
         * Real-ESRGAN session right after this, and it must not be closed while a call is still using it (an
         * ONNX Runtime call cannot be interrupted).
         */
        override fun close() {
            ex.shutdownNow()
            awaitShutdown(ex, CLOSE_LOG_MS) { ms ->
                progress.log("   waiting for ${running.get()} sharpening call(s) to finish before the models are closed (${ms / 1000} s so far)")
            }
        }
    }

    /** the sharpened patch (unwrapping a pool thread's error) */
    private fun java.util.concurrent.Future<Img>.take(): Img = try {
        get()
    } catch (e: java.util.concurrent.ExecutionException) {
        throw (e.cause as? Exception) ?: e
    }

    /**
     * Half-size (centre to edge, processed pixels) a mosaic square needs to be cleaned: --min_mosaic_size (square
     * width), else [MIN_HALF].
     */
    private fun minHalf(): Int = if (opt.minMosaicSize > 0) maxOf(1, opt.minMosaicSize / 2) else MIN_HALF

    /** frame size seen by the last detection pass */
    private var frameW = 0
    private var frameH = 0

    /**
     * Frames of the last video clean that had a mosaic to clean but were written unchanged (a restorer chunk, a patch
     * or the paste failed; each is in the log), so the caller can tell the user instead of only "finished".
     */
    var uncleanedFrames = 0
        private set

    internal companion object {
        const val PATCH_PAD = 10
        /** default cut-off (half-size): 16 = 32 px squares, the restorer's smallest training crop */
        const val MIN_HALF = 16
        /** while the sharpening pool shuts down, a log line this often (ms) says what it is still waiting for */
        const val CLOSE_LOG_MS = 60_000L

        /**
         * Waits until [ex] (already shut down) has finished every task, with no time limit; calls [onWait] with the
         * time waited so far every [logEveryMs] while tasks are still running. An interrupt does not end the wait
         * (it is restored afterwards).
         */
        fun awaitShutdown(ex: java.util.concurrent.ExecutorService, logEveryMs: Long, onWait: (waitedMs: Long) -> Unit) {
            var waited = 0L
            var interrupted = false
            while (true) {
                try {
                    if (ex.awaitTermination(logEveryMs, java.util.concurrent.TimeUnit.MILLISECONDS)) break
                    waited += logEveryMs
                    onWait(waited)
                } catch (e: InterruptedException) {
                    interrupted = true
                }
            }
            if (interrupted) Thread.currentThread().interrupt()
        }
    }

    // ------------------------------------------------------------------ images

    /** [inPlace]: paint the result into [img] instead of a copy (saves a full-size image of memory on big photos). */
    fun cleanMosaicImg(img: Img, netM: MaskNet, restorer: Restorer, inPlace: Boolean = false): Img {
        beginStep("Clean Mosaic")
        val x = t("pre") { netM.input(img) }
        val y = t("inference") { netM.run(x) }
        val pos = t("post") { Nets.mosaicPosFromSeg(img.w, img.h, netM.output(img.w, img.h, y), opt) }
        var result = if (inPlace) img else img.copy()
        if (pos.size > minHalf()) {
            val crop = img.crop(pos.x - pos.size, pos.y - pos.size, pos.size * 2, pos.size * 2)
            val fake = sharpen(t("inference") { restorer.still(crop) }, pos.size)
            result = t("post") { ImageOps.replaceMosaic(result, fake, pos.mask, pos.x, pos.y, pos.size, opt.noFeather) }
        } else {
            progress.log("No mosaic found")
        }
        progress.progress(1, 1)
        endStep(1)
        return result
    }

    // ------------------------------------------------------------------ video

    private class Seg(val w: Int, val h: Int, val t: Net.Tensor)

    /** Pass 1: read -> segment -> [onResult] (in frame order, writer thread). Returns the frame count. */
    private fun segmentPass(open: () -> FrameSource, net: MaskNet, onResult: (w: Int, h: Int, seg: Img) -> Img): Int {
        open().use { src ->
            val c = Counter(src.estimatedFrames)
            staged.run(
                produce = { t("read") { src.next() }?.let { f -> Seg(f.w, f.h, t("pre") { net.input(f) }) } },
                work = { s -> Seg(s.w, s.h, t("inference") { net.run(s.t) }) },
                consume = { s ->
                    val pv = t("post") { onResult(s.w, s.h, net.output(s.w, s.h, s.t)) }
                    progress.preview(pv)
                    c.tick()
                },
                isCancelled = cancelled,
            )
            if (c.done == 0) throw IllegalStateException("No frames could be read from this video.")
            return c.done
        }
    }

    /** [minSize]: half-size a frame needs to be cleaned (see [minHalf]), for the stats lines. */
    private fun findMosaicPositions(open: () -> FrameSource, netM: MaskNet, masks: PackedImages, minSize: Int): Array<IntArray> {
        beginStep("Step:2/4 -- Find mosaic location")
        val xs = ArrayList<Double>(); val ys = ArrayList<Double>(); val ss = ArrayList<Double>()
        var shortSide = 0
        val n = segmentPass(open, netM) { w, h, seg ->
            shortSide = minOf(w, h)
            frameW = w; frameH = h
            val p = Nets.mosaicPosFromSeg(w, h, seg, opt)
            xs.add(p.x.toDouble()); ys.add(p.y.toDouble()); ss.add(p.size.toDouble())
            masks.add(p.mask)
            p.mask
        }
        endStep(n)
        progress.log("Optimize mosaic locations...")
        val fx = Filt.medfilt(xs.toDoubleArray(), opt.medfiltNum)
        val fy = Filt.medfilt(ys.toDoubleArray(), opt.medfiltNum)
        val fs = Filt.medfilt(ss.toDoubleArray(), opt.medfiltNum)
        logDetection(ss, fs, minSize, shortSide)
        return Array(n) { intArrayOf(fx[it].toInt(), fy[it].toInt(), fs[it].toInt()) }
    }

    /** Where detection found mosaic, before and after smoothing, and how often cleaning stops and restarts. */
    private fun logDetection(raw: List<Double>, smoothed: DoubleArray, minSize: Int, shortSide: Int = 0) {
        val n = raw.size
        val found = raw.count { it > 0 }
        val big = raw.count { it > minSize }
        val clean = smoothed.count { it > minSize }
        var runs = 0
        var prev = false
        for (s in smoothed) { val on = s > minSize; if (on && !prev) runs++; prev = on }
        fun pct(k: Int) = String.format(Locale.US, "%.0f%%", 100.0 * k / n.coerceAtLeast(1))
        progress.log(
            "   detection: mosaic found in $found/$n frames (${pct(found)}), large enough in $big (${pct(big)}); " +
                "after smoothing (--medfilt_num ${opt.medfiltNum}) $clean frames (${pct(clean)}) will be cleaned in $runs stretch(es)" +
                if (runs > 1) ", ${runs - 1} gap(s) where frames pass through untouched" else "",
        )
        tooSmallSizes(raw, minSize, shortSide)?.let { progress.log(it) }
        progress.log(detectionTimeline(raw, smoothed, minSize))
    }

    /**
     * How big the mosaics were that detection found but the size cut-off rejected (square width in processed
     * pixels, like --min_mosaic_size), so it shows whether lowering the cut-off would catch them.
     */
    internal fun tooSmallSizes(raw: List<Double>, minSize: Int, shortSide: Int = 0): String? {
        val sizes = raw.filter { it > 0 && it <= minSize }.map { (it * 2).toInt() }.sorted()
        if (sizes.isEmpty()) return null
        val cut = minSize * 2
        val step = maxOf(1, (cut + 3) / 4)
        val buckets = (0 until 4).map { b ->
            val lo = b * step; val hi = minOf(cut, (b + 1) * step)
            "$lo–$hi px: ${sizes.count { it in lo until hi || (b == 3 && it == cut) }}"
        }
        val median = sizes[sizes.size / 2]
        val pct = if (shortSide > 0) String.format(Locale.US, " (%.0f%% of the frame's short side %d px)", 100.0 * median / shortSide, shortSide) else ""
        return "   too small to clean: ${sizes.size} frame(s), mosaic square ${sizes.first()}–${sizes.last()} px, median $median px$pct; " +
            "cut-off ${if (opt.minMosaicSize > 0) "--min_mosaic_size $cut" else "$cut px (default)"}\n" +
            "     sizes: " + buckets.joinToString(" · ")
    }

    /**
     * Which stretches of the video get cleaned and which pass through untouched, with the reason, so a frame that
     * looks unclean can be matched to the log: no mosaic found, found but too small, or dropped by the smoothing.
     * Times are in the result (0 = first processed frame; add -ss for the source time).
     */
    internal fun detectionTimeline(raw: List<Double>, smoothed: DoubleArray, minSize: Int, maxLines: Int = 60): String = buildString {
        val n = raw.size
        fun time(i: Int): String {
            if (frameRate <= 0) return "frame $i"
            val sec = i / frameRate
            val s = sec.toLong()
            return if (s >= 3600) String.format(Locale.US, "%d:%02d:%02d.%d", s / 3600, s / 60 % 60, s % 60, ((sec - s) * 10).toInt())
            else String.format(Locale.US, "%02d:%02d.%d", s / 60, s % 60, ((sec - s) * 10).toInt())
        }
        append("   detection timeline")
        append(if (frameRate > 0) String.format(Locale.US, " (%.2f fps, times in the result%s):", frameRate, if (opt.startTime > 0) "; source time = + ${Options.fmtTime(opt.startTime)}" else "") else " (frame numbers):")
        var lines = 0
        var i = 0
        var runsTotal = 0
        while (i < n) {
            val on = smoothed[i] > minSize
            var j = i
            while (j + 1 < n && (smoothed[j + 1] > minSize) == on) j++
            runsTotal++
            if (lines < maxLines) {
                val len = j - i + 1
                val span = if (frameRate > 0) "${time(i)}–${time(j + 1)}" else "frames $i–$j"
                val detail = if (on) {
                    val bridged = (i..j).count { raw[it] <= minSize }
                    "$len frame(s)" + if (bridged > 0) "; $bridged without a detection, bridged by the smoothing" else ""
                } else {
                    val none = (i..j).count { raw[it] <= 0 }
                    val smallSizes = (i..j).filter { raw[it] > 0 && raw[it] <= minSize }.map { (raw[it] * 2).toInt() }.sorted()
                    val small = smallSizes.size
                    val dropped = len - none - small
                    "$len frame(s): " + listOfNotNull(
                        if (none > 0) "no mosaic found $none" else null,
                        if (small > 0) "mosaic too small $small" + (if (small == 1) " (${smallSizes[0]} px)" else " (${smallSizes.first()}–${smallSizes.last()} px, median ${smallSizes[small / 2]})") else null,
                        if (dropped > 0) "found but dropped by the smoothing $dropped" else null,
                    ).joinToString(", ")
                }
                append(String.format(Locale.US, "\n     %-19s %-9s (%s)", span, if (on) "cleaned" else "untouched", detail))
                lines++
            }
            i = j + 1
        }
        if (runsTotal > lines) append("\n     … ${runsTotal - lines} more stretch(es)")
    }

    /** Pass 2 counts for the log. Updated on the writer thread only (restarts on the AI thread, read after join). */
    private class PassStats {
        var cleaned = 0
        var passed = 0
        var errors = 0
        @Volatile var restarts = 0
        fun line() = "   cleaned $cleaned frame(s), passed $passed through untouched" + (if (errors > 0) ", $errors error(s)" else "")
    }

    // ------------------------------------------------------------------ restorer

    /**
     * Crop squares for the restorer, per frame, steady over each cleaned stretch: the restorer was trained on clips
     * whose crop barely moves, so the size follows the largest mosaic within ±8 frames (never cut off), averaged over
     * ±4 frames (no pumping), and the centre is averaged over ±2 frames on top of the median filter.
     */
    internal class CropPlan(val on: BooleanArray, val x: IntArray, val y: IntArray, val half: IntArray, val stretches: List<IntRange>) {
        /** the stretch each cleaned frame belongs to */
        val stretchOf = IntArray(on.size) { -1 }.also { s -> stretches.forEachIndexed { k, r -> for (i in r) s[i] = k } }
    }

    internal fun planCrops(pos: Array<IntArray>, w: Int, h: Int, minSize: Int): CropPlan {
        val n = pos.size
        val on = BooleanArray(n) { pos[it][2] > minSize }
        val x = IntArray(n); val y = IntArray(n); val half = IntArray(n)
        val stretches = ArrayList<IntRange>()
        val cap = maxOf(1, minOf(w, h) / 2)
        var i = 0
        while (i < n) {
            if (!on[i]) { i++; continue }
            var j = i
            while (j + 1 < n && on[j + 1]) j++
            stretches.add(i..j)
            fun win(k: Int, r: Int) = maxOf(i, k - r)..minOf(j, k + r)
            val grow = IntArray(j - i + 1) { k -> win(i + k, 8).maxOf { pos[it][2] } }
            for (k in i..j) {
                half[k] = kotlin.math.ceil(win(k, 4).map { grow[it - i] }.average()).toInt().coerceIn(1, cap)
                val cx = Math.round(win(k, 2).map { pos[it][0] }.average()).toInt()
                val cy = Math.round(win(k, 2).map { pos[it][1] }.average()).toInt()
                // keep the square inside the frame by moving it, not by shrinking it
                x[k] = cx.coerceIn(half[k], w - half[k])
                y[k] = cy.coerceIn(half[k], h - half[k])
            }
            i = j + 1
        }
        return CropPlan(on, x, y, half, stretches)
    }

    /** [start]..[end] (inclusive) through a [len]-frame restorer call; ramps where it overlaps a neighbouring chunk. */
    internal class ChunkPlan(val start: Int, val end: Int, val len: Int, val rampIn: Boolean, val rampOut: Boolean) {
        fun weight(f: Int): Float {
            var w = 1f
            if (rampIn) w = minOf(w, (f - start + 1f) / (Restorer.OVERLAP + 1))
            if (rampOut) w = minOf(w, (end - f + 1f) / (Restorer.OVERLAP + 1))
            return w
        }
    }

    internal fun planChunks(stretches: List<IntRange>): List<ChunkPlan> {
        val T = Restorer.T
        val out = ArrayList<ChunkPlan>()
        for (r in stretches) {
            val a = r.first; val b = r.last
            val len = b - a + 1
            when {
                len <= Restorer.T_SHORT -> out.add(ChunkPlan(a, b, Restorer.T_SHORT, false, false))
                len <= T -> out.add(ChunkPlan(a, b, T, false, false))
                else -> {
                    var s = a
                    while (s + T - 1 < b) {
                        out.add(ChunkPlan(s, s + T - 1, T, s != a, true))
                        s += T - Restorer.OVERLAP
                    }
                    out.add(ChunkPlan(b - T + 1, b, T, true, false))
                }
            }
        }
        return out
    }

    private class RestoreItem(val index: Int, val chunk: ChunkPlan, val input: Net.Tensor? = null, val error: Exception? = null) {
        var output: Net.Tensor? = null
        var failure: Exception? = null
    }

    /**
     * Detection pass, then the restorer over each cleaned stretch in overlapping 16-frame chunks.
     * The video is read twice at once in pass 2: [open] #1 (reader thread) runs ahead and cuts the crops, [open] #2
     * (writer thread) follows behind and pastes the restored crops into the full frames, so only crops wait in
     * memory, whatever the video's size or length.
     */
    fun cleanMosaicVideoRestorer(open: () -> FrameSource, sink: FrameSink, restorer: Restorer, netM: MaskNet) {
        uncleanedFrames = 0
        val masks = PackedImages()
        val minSize = minHalf()
        val pos = findMosaicPositions(open, netM, masks, minSize)
        val n = pos.size
        beginStep("Step:3/4 -- Clean Mosaic")
        val plan = planCrops(pos, frameW, frameH, minSize)
        val chunks = planChunks(plan.stretches)
        progress.log(
            "   restorer: ${plan.stretches.size} stretch(es), ${chunks.count { it.len == Restorer.T }} chunk(s) of ${Restorer.T} frames " +
                "(${Restorer.OVERLAP} shared and cross-faded), ${chunks.count { it.len == Restorer.T_SHORT }} short chunk(s) of ${Restorer.T_SHORT}",
        )
        val c = Counter(n)
        val stats = PassStats()
        // reader-thread state
        var readIdx = 0
        var chunkIdx = 0
        val crops = HashMap<Int, Img>()
        // AI-thread state: restored crops being cross-faded until no later chunk covers them
        val acc = HashMap<Int, FloatArray>()
        val wsum = HashMap<Int, Float>()
        // finished (and with --enhance_patch sharpened) patches, made on the AI thread or the sharpening pool, pasted by the writer
        val patches = java.util.concurrent.ConcurrentHashMap<Int, java.util.concurrent.Future<Img>>()
        val pool = if (sharpenThreads > 0) SharpenPool() else null
        // writer-thread state
        var follow: FrameSource? = null
        var writeIdx = 0

        fun maskFor(i: Int): Img {
            // union over ±2 frames of the stretch: a frame where detection blinked still gets the neighbours' area
            val r = plan.stretches[plan.stretchOf[i]]
            val m = masks[i]
            for (k in maxOf(r.first, i - 2)..minOf(r.last, i + 2)) {
                if (k == i) continue
                val o = masks[k]
                if (o.w != m.w || o.h != m.h) continue
                for (p in m.data.indices) if (o.data[p].toInt() != 0 && m.data[p].toInt() == 0) m.data[p] = o.data[p]
            }
            return m
        }

        /** Writes frames up to [limit] (exclusive) from the following reader, pasting their restored crops. */
        fun flushTo(limit: Int) {
            val src = follow ?: open().also { follow = it }
            while (writeIdx < minOf(limit, n)) {
                if (progress.isCancelled) throw CancelledException()
                val f = t("read") { src.next() } ?: break
                val i = writeIdx++
                var result = f
                val later = patches.remove(i)
                if (plan.on[i]) {
                    if (later != null) {
                        try {
                            val fake = if (later.isDone) later.take() else t("sharpen_wait") { later.take() }
                            // paste into a copy: a frame source may hand out the same image again (GIF frames, tests)
                            result = t("post") { ImageOps.replaceMosaic(f.copy(), fake, maskFor(i), plan.x[i], plan.y[i], plan.half[i], opt.noFeather) }
                            stats.cleaned++
                        } catch (e: Exception) {
                            progress.log("Warning: frame $i: $e")
                            stats.errors++
                        }
                    } else stats.errors++
                } else stats.passed++
                t("write") { sink.write(result) }
                progress.preview(result)
                c.tick()
            }
        }

        try { open().use { src ->
            staged.run(
                produce = produce@{
                    if (chunkIdx >= chunks.size) return@produce null
                    val k = chunkIdx++
                    val ch = chunks[k]
                    crops.keys.removeAll { it < ch.start }
                    try {
                        while (readIdx <= ch.end) {
                            val f = t("read") { src.next() } ?: break
                            val i = readIdx++
                            if (plan.on[i] && i >= ch.start) {
                                val h = plan.half[i]
                                crops[i] = t("pre") { Restorer.toCrop(f.crop(plan.x[i] - h, plan.y[i] - h, h * 2, h * 2)) }
                            }
                        }
                        val list = (ch.start..ch.end).mapNotNull { crops[it] }
                        if (list.size != ch.end - ch.start + 1) throw IllegalStateException("video ended early (frame $readIdx of $n)")
                        RestoreItem(k, ch, input = t("pre") { Restorer.input(list, ch.len) })
                    } catch (e: Exception) {
                        RestoreItem(k, ch, error = e)
                    }
                },
                work = { it ->
                    val ch = it.chunk
                    if (it.input != null) {
                        try {
                            it.output = t("inference") { restorer.infer(it.input) }
                            stats.restarts++
                        } catch (e: CancelledException) {
                            throw e
                        } catch (e: Exception) {
                            it.failure = e
                        }
                    }
                    val out = it.output
                    if (out != null) {
                        t("post") {
                            for (f in ch.start..ch.end) {
                                val w = ch.weight(f)
                                Restorer.accumulate(out, f - ch.start, w, acc.getOrPut(f) { FloatArray(3 * Nets.CROP * Nets.CROP) })
                                wsum[f] = (wsum[f] ?: 0f) + w
                            }
                        }
                    }
                    // frames no later chunk covers are final: blend and sharpen them here, on the AI thread (the fast
                    // cores), so the writer only pastes
                    val lastFinal = minOf(ch.end, (chunks.getOrNull(it.index + 1)?.start ?: n) - 1)
                    for (f in ch.start..lastFinal) {
                        val a = acc.remove(f)
                        val ws = wsum.remove(f)
                        if (!plan.on[f] || a == null || ws == null || ws <= 0f) continue
                        try {
                            val raw = t("post") { Restorer.toImg(a, ws) }
                            patches[f] = pool?.submit(raw, plan.half[f]) ?: java.util.concurrent.CompletableFuture.completedFuture(sharpen(raw, plan.half[f]))
                        } catch (e: CancelledException) {
                            throw e
                        } catch (e: Exception) {
                            progress.log("Warning: frame $f: $e")
                        }
                    }
                    it
                },
                consume = { it ->
                    val ch = it.chunk
                    if (it.output == null) progress.log("Error: frames ${ch.start}–${ch.end} left uncleaned: ${it.error ?: it.failure}")
                    val next = chunks.getOrNull(it.index + 1)
                    flushTo(next?.start ?: n)
                },
                isCancelled = cancelled,
            )
            flushTo(n)
            if (c.done < n) progress.log("   note: the second read of the video ended at frame ${c.done} of $n")
            endStep(c.done)
            uncleanedFrames = stats.errors
            progress.log(stats.line() + ", ${stats.restarts} restorer call(s)")
            pool?.let { progress.log(it.summary()) }
        } } finally {
            // the pool is closed even if closing the second reader fails: its calls must end before the caller closes the models
            try { follow?.close() } finally { pool?.close() }
        }
    }
}
