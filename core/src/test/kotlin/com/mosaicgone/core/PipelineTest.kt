package com.mosaicgone.core

import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs

/**
 * Runs the pipeline with desktop ONNX Runtime: Real-ESRGAN against the Python RealESRGANer outputs in
 * testdata/reference, and the restorer pipeline's image, video, parallel and log behaviour.
 */
class PipelineTest {

    companion object {
        /** test fixtures, see core/testdata/README.md (paths are relative to the core module, where Gradle runs the tests) */
        val data = File("testdata")
        val models = File("../app/src/main/assets/models")
        val ref = File(data, "reference")
        val gt = File(data, "gt")
        val images = File(data, "images")
        val out = File("build/test-out").apply { mkdirs() }
        private val cache = HashMap<String, Net>()

        fun net(name: String) = cache.getOrPut(name) { Net.fromFile(File(models, "$name.onnx").path) }

        fun maskNet() = MaskNet(net(Options.DETECTOR))

        /** [model]: Options.RESTORER (ours) or Options.LADA */
        fun restorer(video: Boolean = true, model: String = Options.RESTORER) = Restorer(if (video) net(model) else null, net(model + "_t4"))

        @AfterClass @JvmStatic
        fun closeNets() {
            cache.values.forEach { it.close() }
            cache.clear()
        }

        fun load(f: File): Img {
            val bi = ImageIO.read(f) ?: error("cannot read $f")
            val img = Img(bi.width, bi.height, 3)
            for (y in 0 until bi.height) for (x in 0 until bi.width) {
                val rgb = bi.getRGB(x, y)
                img[x, y, 0] = rgb shr 16 and 0xFF; img[x, y, 1] = rgb shr 8 and 0xFF; img[x, y, 2] = rgb and 0xFF
            }
            return img
        }

        fun save(img: Img, name: String) {
            val bi = BufferedImage(img.w, img.h, BufferedImage.TYPE_INT_RGB)
            for (y in 0 until img.h) for (x in 0 until img.w) {
                val v = if (img.c == 1) img[x, y].let { it * 0x10101 } else (img[x, y, 0] shl 16) or (img[x, y, 1] shl 8) or img[x, y, 2]
                bi.setRGB(x, y, v)
            }
            ImageIO.write(bi, "png", File(out, name))
        }

        fun meanAbsDiff(a: Img, b: Img): Double {
            assertEquals("width", a.w, b.w); assertEquals("height", a.h, b.h)
            var s = 0L
            for (i in a.data.indices) s += abs((a.data[i].toInt() and 0xFF) - (b.data[i].toInt() and 0xFF))
            return s.toDouble() / a.data.size
        }

        val quiet = object : Progress {
            override fun log(msg: String) = println("  log: $msg")
        }

        /** a frame with a mosaic: the middle frame of the synthetic "shapes" clip (tools/make_test_clips.py) */
        fun mosaicFrame(): Img = File(gt, "shapes/mosaic").listFiles { f -> f.name.endsWith(".png") }!!
            .sortedBy { it.name }.let { load(it[it.size / 2]) }
    }

    private fun timed(label: String, block: () -> Img): Img {
        val t = System.nanoTime()
        val r = block()
        println("$label: ${(System.nanoTime() - t) / 1_000_000} ms")
        return r
    }

    // ---------------------------------------------------------------- Real-ESRGAN vs the Python RealESRGANer

    private fun maxAbsDiff(a: Img, b: Img): Int {
        var m = 0
        for (i in a.data.indices) m = maxOf(m, kotlin.math.abs((a.data[i].toInt() and 0xFF) - (b.data[i].toInt() and 0xFF)))
        return m
    }

    /** The Real-ESRGAN port (used for the cleaned patch) against the Python RealESRGANer on whole sample images. */
    private fun checkEnhance(model: String, input: String, tile: Int, outscale: Double, refName: String, denoise: Double = 0.5) {
        val img = load(File(ref, "esrgan_input_$input.png"))
        val sr = Enhancer(net(Options(denoiseStrength = denoise).srFile(model)), tile)
        val res = timed("enhance $refName") { sr.enhance(img, outscale) }
        save(res, "esrgan_$refName.png")
        val py = load(File(ref, "esrgan_$refName.png"))
        val d = meanAbsDiff(res, py)
        val m = maxAbsDiff(res, py)
        println("enhance $refName: ${res.w}x${res.h}, mean diff %.3f, max %d".format(d, m))
        assertTrue("Real-ESRGAN differs from Python: mean $d", d < 0.05)
        assertTrue("Real-ESRGAN differs from Python: max $m", m <= 2)
    }

    @Test fun enhanceGeneralX4MatchesPython() = checkEnhance("realesr-general-x4v3", "0014", 0, 4.0, "general050_0014_x4")

    @Test fun enhanceTiledMatchesPython() = checkEnhance("realesr-general-x4v3", "0014", 64, 4.0, "general050_0014_x4_tile64")

    @Test fun enhanceX2LanczosMatchesPython() = checkEnhance("realesr-general-x4v3", "0030", 0, 2.0, "general100_0030_x2", denoise = 1.0)

    @Test fun enhanceAnimeMatchesPython() = checkEnhance("realesr-animevideov3", "ADE_val_00000114", 0, 4.0, "anime_ADE_x4")

    // ---------------------------------------------------------------- images

    @Test fun cleanImage() {
        val img = mosaicFrame()
        val res = timed("clean photo") { Pipeline(Options(modelName = Options.RESTORER).resolve(), quiet).cleanMosaicImg(img, maskNet(), restorer(video = false)) }
        save(res, "photo_clean.png")
        assertTrue("nothing was cleaned", meanAbsDiff(img, res) > 0.05)
    }

    @Test fun cleanWithPatchSharpening() {
        val img = mosaicFrame()
        val opt = Options(modelName = Options.RESTORER, enhancePatch = true).resolve()
        val sr = Enhancer(net(opt.srFile(opt.srModel)))
        val plain = Pipeline(Options(modelName = Options.RESTORER).resolve(), quiet).cleanMosaicImg(img, maskNet(), restorer(video = false))
        val p = Pipeline(opt, quiet, patchSr = sr)
        val sharp = timed("clean + sharpen") { p.cleanMosaicImg(img, maskNet(), restorer(video = false)) }
        save(sharp, "photo_clean_sharpened.png")
        assertTrue("sharpening changed nothing", meanAbsDiff(plain, sharp) > 0.05)
        assertEquals(1L, p.timing.snapshot().getValue("enhance").count)
    }

    // ---------------------------------------------------------------- video

    /** A fake "video": the frame slides a few pixels per frame, with one blank frame. */
    private fun fakeVideo(src: Img, frames: Int) = {
        object : FrameSource {
            var i = 0
            override val estimatedFrames = frames
            override fun next(): Img? {
                if (i >= frames) return null
                val dx = (i * 3) % 24
                val f = Img(src.w - 24, src.h, 3)
                if (i != 7) f.paste(src.crop(dx, 0, src.w - 24, src.h), 0, 0)
                i++
                return f
            }
            override fun close() {}
        }
    }

    private class CollectSink : FrameSink {
        val frames = ArrayList<Img>()
        override fun write(img: Img) { frames.add(img) }
        override fun close() {}
    }

    private fun assertSameFrames(a: List<Img>, b: List<Img>) {
        assertEquals(a.size, b.size)
        for (i in a.indices) assertTrue("frame $i differs", a[i].data.contentEquals(b[i].data))
    }

    @Test fun cleanVideo() {
        val img = mosaicFrame()
        val sink = CollectSink()
        val p = Pipeline(Options(modelName = Options.RESTORER).resolve(), quiet)
        p.cleanMosaicVideoRestorer(fakeVideo(img, 20), sink, restorer(), maskNet())
        assertEquals(20, sink.frames.size)
        assertEquals("frames left uncleaned", 0, p.uncleanedFrames)
        save(sink.frames[3], "video_clean_3.png")
        assertTrue(meanAbsDiff(sink.frames[3], fakeVideo(img, 20)().let { s -> repeat(3) { s.next() }; s.next()!! }) > 0.05)
    }

    /**
     * Every restorer chunk fails (no 16-frame model loaded): the video is still written in full, unchanged, and the
     * frames that should have been cleaned are counted for the caller.
     */
    @Test fun failedRestorerChunksAreCounted() {
        val img = mosaicFrame()
        val sink = CollectSink()
        val p = Pipeline(Options(modelName = Options.RESTORER).resolve(), quiet)
        p.cleanMosaicVideoRestorer(fakeVideo(img, 20), sink, Restorer(null, net(Options.RESTORER + "_t4")), maskNet())
        assertEquals(20, sink.frames.size)
        val src = fakeVideo(img, 20)()
        for (i in 0 until 20) assertTrue("frame $i was changed", sink.frames[i].data.contentEquals(src.next()!!.data))
        assertTrue("no frame counted as uncleaned", p.uncleanedFrames > 0)
        println("uncleaned frames: ${p.uncleanedFrames} of 20")
    }

    /** Lada's restorer through the same pipeline: photo and video clean, parallel == sequential. */
    @Test fun ladaRestorer() {
        val img = mosaicFrame()
        val opt = Options(modelName = Options.LADA).resolve()
        val photo = timed("lada photo") { Pipeline(opt, quiet).cleanMosaicImg(img, maskNet(), restorer(video = false, model = Options.LADA)) }
        save(photo, "lada_photo_clean.png")
        assertTrue("Lada cleaned nothing on the photo", meanAbsDiff(img, photo) > 0.05)
        fun run(parallel: Boolean): List<Img> {
            val sink = CollectSink()
            Pipeline(Options(modelName = Options.LADA, parallel = parallel).resolve(), quiet)
                .cleanMosaicVideoRestorer(fakeVideo(img, 20), sink, restorer(model = Options.LADA), maskNet())
            return sink.frames
        }
        val par = run(true)
        assertEquals(20, par.size)
        save(par[3], "lada_video_clean_3.png")
        assertSameFrames(run(false), par)
    }

    @Test fun parallelWithPatchMatchesSequential() {
        val img = mosaicFrame()
        val sr = Enhancer(net("realesr-animevideov3"))
        var enhanceThread = ""
        fun run(parallel: Boolean): List<Img> {
            val sink = CollectSink()
            val p = Pipeline(Options(modelName = Options.RESTORER, enhancePatch = true, parallel = parallel).resolve(), quiet, patchSr = sr)
            p.cleanMosaicVideoRestorer(fakeVideo(img, 20), sink, restorer(), maskNet())
            enhanceThread = p.timing.snapshot().getValue("enhance").thread
            return sink.frames
        }
        assertSameFrames(run(false), run(true))
        // sharpening runs on the AI thread (the app pins it to the performance cores), not on the writer
        assertEquals("main", enhanceThread)
    }

    /**
     * --parallel_sharpen: the pool gives the same frames as sharpening on the AI thread; with one pool thread the AI
     * thread has to help, still the same frames; the pool's work is timed on the "sharpen" threads.
     */
    @Test fun parallelSharpenPoolMatchesAiThread() {
        val img = mosaicFrame()
        val sr = Enhancer(net("realesr-animevideov3"))
        fun run(parallelSharpen: Boolean, threads: Int): Triple<List<Img>, Pipeline, List<String>> {
            val sink = CollectSink()
            val lines = ArrayList<String>()
            val prog = object : Progress { override fun log(msg: String) { if ("sharpening:" in msg) synchronized(lines) { lines.add(msg.trim()) } } }
            val p = Pipeline(Options(modelName = Options.RESTORER, enhancePatch = true, parallelSharpen = parallelSharpen).resolve(), prog,
                patchSr = sr, sharpenThreads = threads)
            p.cleanMosaicVideoRestorer(fakeVideo(img, 20), sink, restorer(), maskNet())
            return Triple(sink.frames, p, lines)
        }
        val (ai, _, _) = run(false, 0)
        val (pooled, p, lines) = run(true, 3)
        assertSameFrames(ai, pooled)
        println(lines.joinToString())
        val s = p.timing.snapshot().getValue("enhance")
        assertEquals("sharpen", s.thread)
        assertTrue("pool used ${s.threads} thread(s)", s.threads > 1)
        val (one, _, oneLines) = run(true, 1)
        assertSameFrames(ai, one)
        println("1 pool thread: " + oneLines.joinToString())
        assertTrue("AI thread never helped: $oneLines", oneLines.single().let { !it.contains(", 0 on the AI thread") })
    }

    /**
     * The sharpening pool's shutdown waits for a call that ignores interrupts (as an ONNX Runtime call does), however
     * long it takes, and says so while waiting, instead of giving up after a fixed time.
     */
    @Test fun sharpenPoolShutdownWaitsForRunningCall() {
        val ex = java.util.concurrent.Executors.newSingleThreadExecutor()
        val started = java.util.concurrent.CountDownLatch(1)
        val finished = java.util.concurrent.atomic.AtomicBoolean(false)
        ex.submit {
            started.countDown()
            val end = System.nanoTime() + 700_000_000L
            while (System.nanoTime() < end) try { Thread.sleep(20) } catch (_: InterruptedException) {}
            finished.set(true)
        }
        started.await()
        ex.shutdownNow()
        val waits = ArrayList<Long>()
        val t0 = System.nanoTime()
        Pipeline.awaitShutdown(ex, 100) { waits.add(it) }
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertTrue("returned before the running call finished", finished.get())
        assertTrue("returned after $ms ms, the call takes ~700 ms", ms >= 600)
        assertTrue("no wait was reported: $waits", waits.isNotEmpty())
    }

    /**
     * A job cancelled while the pool is sharpening: no sharpening call is still running when the pipeline returns, so
     * the caller can close the Real-ESRGAN session (as JobRunner does) at once.
     */
    @Test fun cancelWithSharpenPoolLeavesNoCallRunning() {
        val img = mosaicFrame()
        val srNet = Net.fromFile(File(models, "realesr-animevideov3.onnx").path, 1)
        // cancel in the cleaning pass (step 3), after a few frames are written, while later patches are on the pool
        val written = java.util.concurrent.atomic.AtomicInteger()
        val cleaning = java.util.concurrent.atomic.AtomicBoolean(false)
        val prog = object : Progress {
            override fun step(text: String) { cleaning.set("Clean Mosaic" in text) }
            override fun progress(done: Int, total: Int) { if (cleaning.get()) written.set(done) }
            override val isCancelled get() = written.get() >= 6
        }
        srNet.use {
            val p = Pipeline(Options(modelName = Options.RESTORER, enhancePatch = true, parallelSharpen = true).resolve(), prog,
                patchSr = Enhancer(srNet), sharpenThreads = 2)
            var cancelled = false
            try {
                p.cleanMosaicVideoRestorer(fakeVideo(img, 40), CollectSink(), restorer(), maskNet())
            } catch (e: CancelledException) {
                cancelled = true
            }
            assertTrue("the job was not cancelled", cancelled)
            val calls = p.timing.snapshot()["enhance"]?.count ?: 0L
            assertTrue("nothing was sharpened before the cancel", calls > 0)
            Thread.sleep(1500)
            // a call still running after the return would finish now and be counted
            assertEquals("a sharpening call was still running after the pipeline returned", calls, p.timing.snapshot()["enhance"]?.count ?: 0L)
        }
    }

    @Test fun detectionTimelineExplainsGaps() {
        val p = Pipeline(Options(modelName = Options.RESTORER).resolve(), quiet, frameRate = 10.0)
        // 0-9 cleaned (one missing detection bridged), 10-14 nothing, 15-16 too small, 17-19 cleaned
        val raw = List(20) { i -> when (i) { 4 -> 0.0; in 10..14 -> 0.0; 15, 16 -> 30.0; else -> 120.0 } }
        val smoothed = DoubleArray(20) { i -> if (i in 10..16) 0.0 else 120.0 }
        val text = p.detectionTimeline(raw, smoothed, 50)
        println(text)
        val lines = text.lines()
        assertEquals(4, lines.size)
        assertTrue(lines[1].contains("00:00.0–00:01.0") && lines[1].contains("cleaned") && lines[1].contains("1 without a detection"))
        assertTrue(lines[2].contains("untouched") && lines[2].contains("no mosaic found 5") && lines[2].contains("mosaic too small 2"))
        assertTrue(lines[3].contains("00:01.7–00:02.0"))
        assertTrue(lines[2].contains("mosaic too small 2 (60–60 px, median 60)"))
        val sizes = p.tooSmallSizes(raw, 50, 480)!!
        println(sizes)
        assertTrue(sizes.contains("2 frame(s), mosaic square 60–60 px, median 60 px (13% of the frame's short side 480 px)"))
        assertTrue(sizes.contains("50–75 px: 2"))
        // --min_mosaic_size 50 (half-size 25) cleans those 60 px frames
        val q = Pipeline(Options(modelName = Options.RESTORER, minMosaicSize = 50).resolve(), quiet, frameRate = 10.0)
        assertEquals(null, q.tooSmallSizes(raw, 25))
    }

    @Test fun timingReportIsLogged() {
        val img = mosaicFrame()
        val lines = ArrayList<String>()
        val prog = object : Progress {
            override fun log(msg: String) { synchronized(lines) { lines.add(msg) } }
        }
        val p = Pipeline(Options(modelName = Options.RESTORER).resolve(), prog)
        p.cleanMosaicVideoRestorer(fakeVideo(img, 10), CollectSink(), restorer(), maskNet())
        val reports = lines.filter { it.startsWith("⏱") }
        reports.forEach { println(it) }
        assertEquals(2, reports.size)
        assertTrue(reports[1].contains("inference") && reports[1].contains("bottleneck"))
        val s = p.timing.snapshot()
        assertEquals("read", s.getValue("read").thread)
        assertEquals("main", s.getValue("inference").thread)
        assertEquals("write", s.getValue("write").thread)
    }
}
