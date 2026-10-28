package com.mosaicgone.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.pow
import kotlin.math.roundToInt

class UnitTest {

    @Test fun medfiltMatchesNumpyVersion() {
        // filt.medfilt(np.array([5,1,9,3,7,2,8.]), 3) -> [1,5,3,7,3,7,2]
        val r = Filt.medfilt(doubleArrayOf(5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0), 3)
        assertArrayEquals(doubleArrayOf(1.0, 5.0, 3.0, 7.0, 3.0, 7.0, 2.0), r, 0.0)
    }

    private fun solid(w: Int, h: Int, r: Int, g: Int, b: Int) = Img(w, h, 3).also { for (p in 0 until w * h) { it.data[p * 3] = r.toByte(); it.data[p * 3 + 1] = g.toByte(); it.data[p * 3 + 2] = b.toByte() } }

    @Test fun yuvKnownValues() {
        // limited range red: BT.601 Y=81 Cb=90 Cr=240, BT.709 Y=63 Cb=102 Cr=240
        val red = solid(4, 2, 255, 0, 0)
        val a = Yuv.fromRgb(red, Yuv.Spec(Yuv.Standard.BT601, false))
        assertEquals(listOf(81, 90, 240), listOf(a[0][0].toInt() and 0xFF, a[1][0].toInt() and 0xFF, a[2][0].toInt() and 0xFF))
        val b = Yuv.fromRgb(red, Yuv.Spec(Yuv.Standard.BT709, false))
        assertEquals(listOf(63, 102, 240), listOf(b[0][0].toInt() and 0xFF, b[1][0].toInt() and 0xFF, b[2][0].toInt() and 0xFF))
        // full range white / black
        val w = Yuv.fromRgb(solid(2, 2, 255, 255, 255), Yuv.Spec(Yuv.Standard.BT709, true))
        assertEquals(255, w[0][0].toInt() and 0xFF)
    }

    @Test fun yuvRoundTrip() {
        val img = PipelineTest.load(java.io.File(PipelineTest.ref, "esrgan_input_ADE_val_00000114.png"))
        for (std in Yuv.Standard.values()) for (full in listOf(false, true)) {
            val spec = Yuv.Spec(std, full)
            val p = Yuv.fromRgb(img, spec)
            val back = Yuv.toRgb(Yuv.Plane(p[0], img.w, 1), Yuv.Plane(p[1], img.w / 2, 1), Yuv.Plane(p[2], img.w / 2, 1), 0, 0, img.w, img.h, spec)
            var d = 0L
            for (i in img.data.indices) d += kotlin.math.abs((img.data[i].toInt() and 0xFF) - (back.data[i].toInt() and 0xFF))
            val mean = d.toDouble() / img.data.size
            println("YUV round trip $spec: mean abs error %.2f".format(mean))
            assertTrue("$spec round trip error $mean", mean < 2.5)
        }
        // a wrong range assumption is what crushes blacks/whites: decoding full-range data as limited changes the picture a lot
        val p = Yuv.fromRgb(img, Yuv.Spec(Yuv.Standard.BT601, true))
        val wrong = Yuv.toRgb(Yuv.Plane(p[0], img.w, 1), Yuv.Plane(p[1], img.w / 2, 1), Yuv.Plane(p[2], img.w / 2, 1), 0, 0, img.w, img.h, Yuv.Spec(Yuv.Standard.BT601, false))
        var d = 0L
        for (i in img.data.indices) d += kotlin.math.abs((img.data[i].toInt() and 0xFF) - (wrong.data[i].toInt() and 0xFF))
        assertTrue(d.toDouble() / img.data.size > 5)
    }

    /** 8-bit planes -> P010 style (16-bit little-endian, value in the top 10 bits), interleaved CbCr like Android's P010 Image */
    private fun p010(p: Array<ByteArray>, w: Int, h: Int): Triple<Yuv.Plane, Yuv.Plane, Yuv.Plane> {
        fun s16(v: Int) = (v * 4) shl 6
        val y = ByteArray(w * h * 2)
        for (i in 0 until w * h) { val v = s16(p[0][i].toInt() and 0xFF); y[2 * i] = v.toByte(); y[2 * i + 1] = (v shr 8).toByte() }
        val cw = w / 2; val ch = h / 2
        val uv = ByteArray(cw * ch * 4)
        for (i in 0 until cw * ch) {
            val u = s16(p[1][i].toInt() and 0xFF); val v = s16(p[2][i].toInt() and 0xFF)
            uv[4 * i] = u.toByte(); uv[4 * i + 1] = (u shr 8).toByte(); uv[4 * i + 2] = v.toByte(); uv[4 * i + 3] = (v shr 8).toByte()
        }
        return Triple(Yuv.Plane(y, w * 2, 2, wide = true), Yuv.Plane(uv, cw * 4, 4, 0, wide = true), Yuv.Plane(uv, cw * 4, 4, 2, wide = true))
    }

    @Test fun tenBitDecodesLikeEightBit() {
        val img = PipelineTest.load(java.io.File(PipelineTest.ref, "esrgan_input_ADE_val_00000114.png"))
        val spec = Yuv.Spec(Yuv.Standard.BT709, false)
        val p = Yuv.fromRgb(img, spec)
        val a = Yuv.toRgb(Yuv.Plane(p[0], img.w, 1), Yuv.Plane(p[1], img.w / 2, 1), Yuv.Plane(p[2], img.w / 2, 1), 0, 0, img.w, img.h, spec)
        val (y, u, v) = p010(p, img.w, img.h)
        val b = Yuv.toRgb(y, u, v, 0, 0, img.w, img.h, spec)
        assertArrayEquals(a.data, b.data)
    }

    @Test fun parallelColorIsIdentical() {
        val img = PipelineTest.load(java.io.File(PipelineTest.ref, "esrgan_input_ADE_val_00000114.png"))
        for (spec in listOf(Yuv.Spec(Yuv.Standard.BT709, false), Yuv.Spec(Yuv.Standard.BT2020, false, Yuv.Transfer.PQ))) {
            RowPool(3).use { pool ->
                val sdr = spec.sdr
                val a = Yuv.fromRgb(img, sdr)
                val b = Yuv.fromRgb(img, sdr, pool)
                for (k in 0..2) assertArrayEquals(a[k], b[k])
                val planes = { Triple(Yuv.Plane(a[0], img.w, 1), Yuv.Plane(a[1], img.w / 2, 1), Yuv.Plane(a[2], img.w / 2, 1)) }
                val (y1, u1, v1) = planes()
                val r1 = Yuv.toRgb(y1, u1, v1, 0, 0, img.w, img.h, spec)
                val (y2, u2, v2) = planes()
                val r2 = Yuv.toRgb(y2, u2, v2, 0, 0, img.w, img.h, spec, rows = pool)
                assertArrayEquals(r1.data, r2.data)
            }
        }
    }

    @Test fun chroma444And422() {
        // a flat colour must decode the same whatever the chroma layout
        val img = solid(8, 4, 200, 60, 30)
        val spec = Yuv.Spec(Yuv.Standard.BT601, false)
        val p = Yuv.fromRgb(img, spec)
        val yv = p[0]; val u0 = p[1][0]; val v0 = p[2][0]
        val ref = Yuv.toRgb(Yuv.Plane(yv, 8, 1), Yuv.Plane(p[1], 4, 1), Yuv.Plane(p[2], 4, 1), 0, 0, 8, 4, spec)
        val full = Yuv.toRgb(Yuv.Plane(yv, 8, 1), Yuv.Plane(ByteArray(32) { u0 }, 8, 1), Yuv.Plane(ByteArray(32) { v0 }, 8, 1), 0, 0, 8, 4, spec, 1, 1)
        val half = Yuv.toRgb(Yuv.Plane(yv, 8, 1), Yuv.Plane(ByteArray(16) { u0 }, 4, 1), Yuv.Plane(ByteArray(16) { v0 }, 4, 1), 0, 0, 8, 4, spec, 2, 1)
        assertArrayEquals(ref.data, full.data)
        assertArrayEquals(ref.data, half.data)
    }

    /** decodes one grey YCbCr value (10-bit limited range code [y10]) with [spec] */
    private fun greyHdr(y10: Int, spec: Yuv.Spec): Int {
        val v = y10 shl 6
        val y = ByteArray(8) { if (it % 2 == 0) v.toByte() else (v shr 8).toByte() }
        val c = (512 shl 6)
        val uv = ByteArray(8) { if (it % 2 == 0) c.toByte() else (c shr 8).toByte() }
        val out = Yuv.toRgb(Yuv.Plane(y, 4, 2, wide = true), Yuv.Plane(uv, 4, 4, 0, true), Yuv.Plane(uv, 4, 4, 2, true), 0, 0, 2, 2, spec)
        assertEquals("grey stays grey", out[0, 0, 0], out[0, 0, 1])
        assertEquals("grey stays grey", out[0, 0, 0], out[0, 0, 2])
        return out[0, 0, 0]
    }

    @Test fun hdrToneMapping() {
        val pq = Yuv.Spec(Yuv.Standard.BT2020, false, Yuv.Transfer.PQ)
        val hlg = Yuv.Spec(Yuv.Standard.BT2020, false, Yuv.Transfer.HLG)
        fun code(signal: Double) = (64 + 876 * signal).roundToInt()
        // PQ 0.58 = 203 nits and HLG 0.75 = 203 nits (on a 1000-nit display): both are HDR reference white
        val pqWhite = greyHdr(code(0.5806), pq)
        val hlgWhite = greyHdr(code(0.75), hlg)
        println("HDR reference white -> SDR $pqWhite (PQ), $hlgWhite (HLG)")
        assertTrue(pqWhite in 236..250)
        assertTrue(kotlin.math.abs(pqWhite - hlgWhite) <= 2)
        assertEquals(0, greyHdr(64, pq))
        assertEquals(255, greyHdr(code(0.7518), pq)) // 1000 nits = the assumed peak
        // 18% grey (36.5 nits) keeps its SDR value: 0.18^(1/2.4) * 255 = 125
        assertTrue(greyHdr(code(pqSignal(0.18 * 203)), pq) in 123..127)
        // brighter in, brighter out
        var last = -1
        for (s in 0..100) { val g = greyHdr(code(s / 100.0 * 0.76), pq); assertTrue(g >= last); last = g }
    }

    /** inverse PQ: nits -> signal */
    private fun pqSignal(nits: Double): Double {
        val m1 = 2610.0 / 16384; val m2 = 2523.0 / 4096 * 128
        val c1 = 3424.0 / 4096; val c2 = 2413.0 / 4096 * 32; val c3 = 2392.0 / 4096 * 32
        val y = (nits / 10000).pow(m1)
        return ((c1 + c2 * y) / (1 + c3 * y)).pow(m2)
    }

    @Test fun rotateMatchesDisplayRotation() {
        val m = Img(3, 2, 1, byteArrayOf(1, 2, 3, 4, 5, 6))
        // 90 clockwise: rows become [4,1],[5,2],[6,3]
        assertArrayEquals(byteArrayOf(4, 1, 5, 2, 6, 3), Yuv.rotate(m, 90).data)
        assertArrayEquals(byteArrayOf(6, 5, 4, 3, 2, 1), Yuv.rotate(m, 180).data)
        assertArrayEquals(byteArrayOf(3, 6, 2, 5, 1, 4), Yuv.rotate(m, 270).data)
    }

    @Test fun areaMatchesOpenCv() {
        // cv2.resize(np.array([[0,100,200,50,10,250,30,90,77,180]],np.uint8),(w,1),interpolation=cv2.INTER_AREA)
        val m = Img(10, 1, 1, byteArrayOf(0, 100, 200.toByte(), 50, 10, 250.toByte(), 30, 90, 77, 180.toByte()))
        assertArrayEquals(intArrayOf(95, 94, 107), ImageOps.resizeArea(m, 3, 1).let { r -> IntArray(3) { r[it, 0] } })
        assertArrayEquals(intArrayOf(50, 125, 130, 60, 128), ImageOps.resizeArea(m, 5, 1).let { r -> IntArray(5) { r[it, 0] } })
        assertArrayEquals(intArrayOf(80, 64, 130, 121), ImageOps.resizeArea(m, 4, 1).let { r -> IntArray(4) { r[it, 0] } })
        // cv2.resize(np.arange(36,dtype=np.uint8).reshape(6,6)*7,(4,4),interpolation=cv2.INTER_AREA)
        val b = Img(6, 6, 1, ByteArray(36) { (it * 7).toByte() })
        val r = ImageOps.resizeArea(b, 4, 4)
        val expect = intArrayOf(16, 26, 37, 47, 72, 82, 93, 103, 142, 152, 163, 173, 198, 208, 219, 229)
        assertArrayEquals(expect, IntArray(16) { r[it % 4, it / 4] })
    }

    @Test fun lanczos4MatchesOpenCv() {
        // cv2.resize(np.array([[0,100,200,50,10,250,30,90]],np.uint8),(w,1),interpolation=cv2.INTER_LANCZOS4)
        val m = Img(8, 1, 1, byteArrayOf(0, 100, 200.toByte(), 50, 10, 250.toByte(), 30, 90))
        val down = ImageOps.resize(m, 4, 1, ImageOps.Interp.LANCZOS4)
        assertArrayEquals(intArrayOf(32, 151, 162, 22), IntArray(4) { down[it, 0] })
        val up = ImageOps.resize(m, 12, 1, ImageOps.Interp.LANCZOS4)
        assertArrayEquals(intArrayOf(0, 32, 127, 200, 151, 16, 0, 162, 238, 65, 22, 105), IntArray(12) { up[it, 0] })
    }

    @Test fun blurMatchesOpenCv() {
        // cv2.blur(np.array([[0,0,255,0,0]],np.uint8),(3,1)) -> [0,85,85,85,0]
        val m = Img(5, 1, 1, byteArrayOf(0, 0, -1, 0, 0))
        val b = ImageOps.blur(m, 3, 1)
        assertArrayEquals(intArrayOf(0, 85, 85, 85, 0), IntArray(5) { b[it, 0] })
    }

    @Test fun resizeIdentityAndUpscale() {
        val m = Img(2, 1, 1, byteArrayOf(0, 100))
        val r = ImageOps.resize(m, 4, 1)
        // cv2.resize(np.array([[0,100]],np.uint8),(4,1)) -> [0,25,75,100]
        assertArrayEquals(intArrayOf(0, 25, 75, 100), IntArray(4) { r[it, 0] })
    }

    @Test fun boundingSquareCentred() {
        val m = Img(100, 100, 1)
        for (y in 40 until 60) for (x in 30 until 50) m[x, y, 0] = 255
        val b = ImageOps.boundingSquare(m, 1.0)
        assertEquals(40, b[0]); assertEquals(50, b[1]); assertEquals(10, b[2])
    }

    @Test fun roiFillsHoles() {
        val m = Img(20, 20, 1)
        for (y in 2 until 12) for (x in 2 until 12) if (x == 2 || x == 11 || y == 2 || y == 11) m[x, y, 0] = 255
        m[17, 17, 0] = 255
        val r = ImageOps.findMostLikelyROI(m)
        assertEquals(255, r[6, 6]); assertEquals(0, r[17, 17])
    }

    @Test fun optionsAutoMode() {
        assertEquals("clean", Options(modelName = Options.RESTORER).resolve().mode)
        assertEquals("clean", Options(modelName = Options.LADA).resolve().mode)
        assertEquals(1.5, Options(modelName = Options.RESTORER).resolve().exMultValue, 0.0)
        assertEquals(false, Options().enhancePatch) // sharpening is off by default
        assertThrows(OptionError::class.java) { Options().resolve() } // no model chosen
        assertThrows(OptionError::class.java) { Options(modelName = "no_such_model").resolve() } // only the shipped models
        assertThrows(OptionError::class.java) { Options(mode = "style", modelName = Options.RESTORER).resolve() }
        assertThrows(OptionError::class.java) { Options(modelName = "realesr-general-x4v3").resolve() }
    }

    @Test fun denoiseLevelsSnapToBundledModels() {
        assertEquals("realesr-general-x4v3_dn050", Options().srFile("realesr-general-x4v3"))
        assertEquals("realesr-general-x4v3_dn000", Options(denoiseStrength = 0.2).srFile("realesr-general-x4v3"))
        assertEquals("realesr-general-x4v3_dn100", Options(denoiseStrength = 0.8).srFile("realesr-general-x4v3"))
        assertEquals("realesr-animevideov3", Options(denoiseStrength = 0.0).srFile("realesr-animevideov3"))
    }

    @Test fun optionsParse() {
        val o = Options()
        Options.applyArgs(o, "--no_feather -ss 00:01:05 -t 10 --medfilt_num 7")
        assertEquals(true, o.noFeather)
        assertEquals(65.0, o.startTime, 0.0); assertEquals(10.0, o.lastTime, 0.0); assertEquals(7, o.medfiltNum)
        val p = Options()
        Options.applyArgs(p, "--threads 3 --no_affinity --no_parallel")
        assertEquals(3, p.threads); assertEquals(false, p.affinity); assertEquals(false, p.parallel)
        assertTrue(p.toCommandLine("x").contains("--threads 3 --no_affinity --no_parallel"))
        val e = Options()
        assertEquals("realesr-animevideov3", e.srModel)
        Options.applyArgs(e, "-dn 0 --enhance_patch --sr_model realesr-general-x4v3")
        assertEquals(0.0, e.denoiseStrength, 0.0)
        assertEquals(true, e.enhancePatch); assertEquals("realesr-general-x4v3", e.srModel)
        assertTrue(e.toCommandLine("x").contains("--denoise_strength 0 --enhance_patch --sr_model realesr-general-x4v3"))
        assertThrows(OptionError::class.java) { Options.applyArgs(Options(), "--traditional") } // removed with the old models
        assertEquals(false, Options().also { Options.applyArgs(it, "--no_parallel_sharpen") }.parallelSharpen)
        assertThrows(OptionError::class.java) { Options.applyArgs(Options(), "--outscale 2") }
        assertThrows(OptionError::class.java) { Options(modelName = Options.RESTORER, threads = -1).resolve() }
        assertThrows(OptionError::class.java) { Options.applyArgs(Options(), "--bogus 1") }
        assertThrows(OptionError::class.java) { Options.applyArgs(Options(), "--fps abc") }
    }

    @Test fun maskStoreRoundTrip() {
        val s = PackedImages()
        val m = Img(37, 21, 1)
        for (i in 0 until 200) m.data[i * 3] = -1
        s.add(m)
        assertArrayEquals(m.data, s[0].data)
    }
}
