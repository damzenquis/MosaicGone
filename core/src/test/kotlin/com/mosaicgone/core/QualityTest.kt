package com.mosaicgone.core

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.log10

/**
 * Ground-truth checks on the clips in testdata/gt/<clip>/{gt,mosaic} (testdata/README.md): PSNR / SSIM of
 * the whole pipeline and its options against the clean original, in the mosaic area only, flicker and false alarms.
 * Only the "shapes" clip is in Git; a test that needs only the other clips is skipped where they are missing.
 */
class QualityTest {

    private class Clip(val name: String, val gt: List<Img>, val mz: List<Img>)

    private fun loadDir(d: File) = d.listFiles { f -> f.name.endsWith(".png") }!!.sortedBy { it.name }.map { PipelineTest.load(it) }

    private fun clip(name: String): Clip? {
        val d = File(PipelineTest.gt, name)
        if (!File(d, "mosaic").isDirectory) {
            println("skip $name: ${d.path} is missing (not in Git, see testdata/README.md)"); return null
        }
        return Clip(name, loadDir(File(d, "gt")), loadDir(File(d, "mosaic")))
    }

    private fun source(frames: List<Img>) = {
        object : FrameSource {
            var i = 0
            override val estimatedFrames = frames.size
            override fun next(): Img? = if (i < frames.size) frames[i++] else null
            override fun close() {}
        }
    }

    private class Collect : FrameSink {
        val frames = ArrayList<Img>()
        override fun write(img: Img) { frames.add(img) }
        override fun close() {}
    }

    private fun mosaicGone(c: Clip, opt: Options = Options(modelName = Options.RESTORER), frames: List<Img> = c.mz, prog: Progress = PipelineTest.quiet,
                           patch: Enhancer? = null): List<Img> {
        val sink = Collect()
        Pipeline(opt.resolve(), prog, patchSr = patch)
            .cleanMosaicVideoRestorer(source(frames), sink, PipelineTest.restorer(model = opt.modelName), PipelineTest.maskNet())
        return sink.frames
    }

    // ---------------------------------------------------------------- metrics

    private fun gray(img: Img, x0: Int, y0: Int, x1: Int, y1: Int): Array<DoubleArray> = Array(y1 - y0) { y ->
        DoubleArray(x1 - x0) { x ->
            val px = x0 + x; val py = y0 + y
            // cv2.COLOR_BGR2GRAY fixed point
            ((img[px, py, 0] * 4899 + img[px, py, 1] * 9617 + img[px, py, 2] * 1868 + 8192) shr 14).toDouble()
        }
    }

    private val kernel: DoubleArray = DoubleArray(11) { exp(-((it - 5) * (it - 5)) / (2 * 1.5 * 1.5)) }.let { k -> val s = k.sum(); DoubleArray(11) { k[it] / s } }

    private fun refl(i: Int, n: Int): Int {
        if (n == 1) return 0
        var p = i
        while (p < 0 || p >= n) p = if (p < 0) -p else 2 * n - 2 - p
        return p
    }

    private fun gauss(a: Array<DoubleArray>): Array<DoubleArray> {
        val h = a.size; val w = a[0].size
        val t = Array(h) { y -> DoubleArray(w) { x -> (0 until 11).sumOf { k -> kernel[k] * a[y][refl(x + k - 5, w)] } } }
        return Array(h) { y -> DoubleArray(w) { x -> (0 until 11).sumOf { k -> kernel[k] * t[refl(y + k - 5, h)][x] } } }
    }

    private fun ssim(a: Array<DoubleArray>, b: Array<DoubleArray>): Double {
        val c1 = (0.01 * 255) * (0.01 * 255); val c2 = (0.03 * 255) * (0.03 * 255)
        val h = a.size; val w = a[0].size
        val ma = gauss(a); val mb = gauss(b)
        val aa = gauss(Array(h) { y -> DoubleArray(w) { x -> a[y][x] * a[y][x] } })
        val bb = gauss(Array(h) { y -> DoubleArray(w) { x -> b[y][x] * b[y][x] } })
        val ab = gauss(Array(h) { y -> DoubleArray(w) { x -> a[y][x] * b[y][x] } })
        var s = 0.0
        for (y in 0 until h) for (x in 0 until w) {
            val sa = aa[y][x] - ma[y][x] * ma[y][x]; val sb = bb[y][x] - mb[y][x] * mb[y][x]; val sab = ab[y][x] - ma[y][x] * mb[y][x]
            s += ((2 * ma[y][x] * mb[y][x] + c1) * (2 * sab + c2)) / ((ma[y][x] * ma[y][x] + mb[y][x] * mb[y][x] + c1) * (sa + sb + c2))
        }
        return s / (w * h)
    }

    private fun psnr(a: Img, b: Img, x0: Int, y0: Int, x1: Int, y1: Int): Double {
        var se = 0.0; var n = 0
        for (y in y0 until y1) for (x in x0 until x1) for (ch in 0 until 3) { val d = (a[x, y, ch] - b[x, y, ch]).toDouble(); se += d * d; n++ }
        val mse = se / n
        return if (mse < 1e-10) 100.0 else 10 * log10(255.0 * 255.0 / mse)
    }

    /** bounding box of where the mosaic changed the frame (threshold 12, 3x3 opening), null if tiny */
    private fun region(gt: Img, mz: Img): IntArray? {
        val w = gt.w; val h = gt.h
        val d = BooleanArray(w * h) { p -> (0 until 3).maxOf { abs((gt.data[p * 3 + it].toInt() and 0xFF) - (mz.data[p * 3 + it].toInt() and 0xFF)) } > 12 }
        fun at(a: BooleanArray, x: Int, y: Int, out: Boolean) = if (x < 0 || y < 0 || x >= w || y >= h) out else a[y * w + x]
        val er = BooleanArray(w * h) { p -> val x = p % w; val y = p / w; (-1..1).all { dy -> (-1..1).all { dx -> at(d, x + dx, y + dy, true) } } }
        val op = BooleanArray(w * h) { p -> val x = p % w; val y = p / w; (-1..1).any { dy -> (-1..1).any { dx -> at(er, x + dx, y + dy, false) } } }
        var x0 = w; var y0 = h; var x1 = -1; var y1 = -1; var n = 0
        for (p in op.indices) if (op[p]) { n++; val x = p % w; val y = p / w; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y }
        return if (n < 50) null else intArrayOf(x0, y0, x1 + 1, y1 + 1)
    }

    private class Score(val psnr: Double, val ssim: Double, val frames: Int)

    private val regions = HashMap<String, List<IntArray?>>()

    private fun score(c: Clip, out: List<Img>): Score {
        val rs = regions.getOrPut(c.name) { c.gt.indices.map { region(c.gt[it], c.mz[it]) } }
        var p = 0.0; var s = 0.0; var n = 0
        for (i in c.gt.indices) {
            val r = rs[i] ?: continue
            p += psnr(c.gt[i], out[i], r[0], r[1], r[2], r[3])
            s += ssim(gray(c.gt[i], r[0], r[1], r[2], r[3]), gray(out[i], r[0], r[1], r[2], r[3]))
            n++
        }
        return Score(p / n, s / n, n)
    }

    /** mean |frame t - frame t-1| in the mosaic area, minus the same for the clean original (0 = as steady as the source) */
    private fun flicker(c: Clip, out: List<Img>): Double {
        val rs = regions.getOrPut(c.name) { c.gt.indices.map { region(c.gt[it], c.mz[it]) } }
        var d = 0.0; var n = 0
        for (i in 1 until out.size) {
            val r = rs[i] ?: continue
            if (rs[i - 1] == null) continue
            var a = 0.0; var b = 0.0; var k = 0
            for (y in r[1] until r[3]) for (x in r[0] until r[2]) for (ch in 0 until 3) {
                a += abs(out[i][x, y, ch] - out[i - 1][x, y, ch]); b += abs(c.gt[i][x, y, ch] - c.gt[i - 1][x, y, ch]); k++
            }
            d += (a - b) / k; n++
        }
        return if (n == 0) 0.0 else d / n
    }

    // ---------------------------------------------------------------- tests

    /**
     * The app's whole pipeline on every ground-truth clip: quality in the mosaic area, flicker, and false alarms on
     * the clean frames (they must pass through untouched). -> build/mosaicgone_quality.md
     */
    @Test fun mosaicGoneQuality() {
        val report = StringBuilder("# MosaicGone, whole app pipeline on the ground-truth clips\n\n" +
            "PSNR (dB) / SSIM in the mosaic area against the clean original; flicker = mean frame-to-frame change in that area " +
            "minus the original's (lower = steadier); false alarms = clean frames (no mosaic) that the pipeline changed.\n\n" +
            "| clip | pipeline | PSNR | SSIM | flicker | false alarms | ms/frame (PC) |\n|---|---|---|---|---|---|---|\n")
        for (name in listOf("real", "shapes", "onepiece", "small", "mid", "tiny")) {
            val c = clip(name) ?: continue
            val runs = listOf<Pair<String, (List<Img>) -> List<Img>>>(
                "default" to { f -> mosaicGone(c, frames = f) },
                "Lada restorer" to { f -> mosaicGone(c, Options(modelName = Options.LADA), frames = f) },
                "+ sharpen realesr-animevideov3" to { f ->
                    mosaicGone(c, Options(modelName = Options.RESTORER, enhancePatch = true), frames = f, patch = Enhancer(PipelineTest.net(Options.SR_ANIME)))
                },
            )
            for ((label, run) in runs) {
                val t0 = System.nanoTime()
                val out = run(c.mz)
                val ms = (System.nanoTime() - t0) / 1e6 / c.mz.size
                assertTrue("$name $label: ${out.size} frames out of ${c.mz.size}", out.size == c.mz.size)
                val s = score(c, out)
                val fl = flicker(c, out)
                val clean = run(c.gt)
                val fa = c.gt.indices.count { !clean[it].data.contentEquals(c.gt[it].data) }
                val line = String.format(Locale.US, "| %s | %s | %.2f | %.4f | %+.2f | %d/%d | %.0f |", name, label, s.psnr, s.ssim, fl, fa, c.gt.size, ms)
                println(line)
                report.append(line).append('\n')
                // tiny: most of its mosaics are below the 32 px cut-off, so it is reported only
                if (name != "tiny") assertTrue("$name $label: no cleaner than the mosaic", s.psnr > score(c, c.mz).psnr)
                assertTrue("$name $label: $fa clean frames changed", fa == 0)
            }
        }
        File("build/mosaicgone_quality.md").writeText(report.toString())
    }

    /** The restorer pipeline gives the same frames with its three threads as all on one thread, and leaves the rest of the frame alone. */
    @Test fun mosaicGoneParallelMatchesSerialAndKeepsOutside() {
        val c = clip("small")
        assumeTrue("the small clip is missing (not in Git, see testdata/README.md)", c != null)
        c!!
        val logs = ArrayList<String>()
        val prog = object : Progress { override fun log(msg: String) { synchronized(logs) { logs.add(msg) }; println("  log: $msg") } }
        val par = mosaicGone(c, prog = prog)
        val ser = mosaicGone(c, Options(modelName = Options.RESTORER, parallel = false))
        for (i in par.indices) assertTrue("frame $i differs between parallel and serial", par[i].data.contentEquals(ser[i].data))
        // pixels far from the mosaic (outside its region grown by its size and the mask dilation) are untouched,
        // and the input frames themselves were not written into
        val again = clip("small")!!
        for (i in c.mz.indices) assertTrue("input frame $i was modified", c.mz[i].data.contentEquals(again.mz[i].data))
        val rs = c.gt.indices.map { region(c.gt[it], c.mz[it]) }
        var touched = 0L; var far = 0L
        for (i in par.indices) {
            val r = rs[i] ?: continue
            // the mask is dilated by design (blur of min(w, h) / 20 at the 360 px mask scale, then feathered): allow that
            val dil = minOf(par[i].w, par[i].h) / 8
            val gw = (r[2] - r[0]) + dil; val gh = (r[3] - r[1]) + dil
            for (y in 0 until par[i].h) for (x in 0 until par[i].w) {
                if (x in (r[0] - gw)..(r[2] + gw) && y in (r[1] - gh)..(r[3] + gh)) continue
                far++
                if ((0 until 3).any { par[i][x, y, it] != c.mz[i][x, y, it] }) touched++
            }
        }
        println("outside the mosaic surroundings: $touched of $far pixels changed")
        assertTrue("nothing to check outside the mosaic", far > 0)
        assertTrue("restorer changed $touched pixels far from the mosaic", touched == 0L)
        assertTrue("no restorer calls logged", logs.any { it.contains("restorer call") })
    }

    /** A photo through the restorer path (4-frame export): cleaner than the mosaic. */
    @Test fun mosaicGonePhoto() {
        for (name in listOf("shapes", "onepiece")) {
            val c = clip(name) ?: continue
            val rs = c.gt.indices.map { region(c.gt[it], c.mz[it]) }
            val i = rs.indexOfFirst { it != null }.takeIf { it >= 0 } ?: continue
            val r = rs[i]!!
            val mg = Pipeline(Options(modelName = Options.RESTORER).resolve(), PipelineTest.quiet)
                .cleanMosaicImg(c.mz[i].copy(), PipelineTest.maskNet(), PipelineTest.restorer(video = false))
            val pm = psnr(c.gt[i], c.mz[i], r[0], r[1], r[2], r[3])
            val pg = psnr(c.gt[i], mg, r[0], r[1], r[2], r[3])
            println(String.format(Locale.US, "photo %s frame %d: mosaic %.2f dB · MosaicGone %.2f dB", name, i, pm, pg))
            PipelineTest.save(mg, "mosaicgone_photo_$name.png")
            assertTrue("$name: photo is no cleaner than the mosaic", pg > pm)
        }
    }

    /** Each More Options setting against the default, on two clips. -> build/quality_report.md */
    @Test fun scoreVariants() {
        val report = StringBuilder("# Mosaic-area quality vs ground truth\n\nPSNR (dB, higher = closer to the original) and SSIM (1 = identical), mosaic area only.\n\n")
        for (name in listOf("onepiece", "shapes")) {
            val c = clip(name) ?: continue
            val base = Options(modelName = Options.RESTORER)
            val variants = listOf<Pair<String, () -> List<Img>>>(
                "mosaic input (no cleaning)" to { c.mz },
                "baseline: default" to { mosaicGone(c, base.copy()) },
                "+ sharpen realesr-general dn 0.5" to { mosaicGone(c, base.copy(enhancePatch = true), patch = Enhancer(PipelineTest.net("realesr-general-x4v3_dn050"))) },
                "--no_feather" to { mosaicGone(c, base.copy(noFeather = true)) },
                "--medfilt_num 5" to { mosaicGone(c, base.copy(medfiltNum = 5)) },
                "--mask_threshold 64" to { mosaicGone(c, base.copy(maskThreshold = 64)) },
                "--all_mosaic_area" to { mosaicGone(c, base.copy(allMosaicArea = true)) },
            )
            report.append("## $name (${c.gt.size} frames, ${c.gt[0].w}x${c.gt[0].h})\n\n| variant | PSNR | SSIM | vs baseline |\n|---|---|---|---|\n")
            var b: Score? = null
            for ((label, run) in variants) {
                val t0 = System.nanoTime()
                val s = score(c, run())
                if (label.startsWith("baseline")) b = s
                val delta = b?.let { String.format(Locale.US, "%+.2f dB / %+.4f", s.psnr - it.psnr, s.ssim - it.ssim) } ?: ""
                val line = String.format(Locale.US, "| %s | %.2f | %.4f | %s |", label, s.psnr, s.ssim, if (label.startsWith("baseline")) "" else delta)
                println("$name $line  (${(System.nanoTime() - t0) / 1_000_000} ms)")
                report.append(line).append('\n')
            }
            report.append('\n')
        }
        File("build/quality_report.md").writeText(report.toString())
    }
}
