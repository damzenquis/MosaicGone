package com.mosaicgone.core

import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * YCbCr <-> RGB with the source's colour standard and range, so decoding, the models and the encoder
 * all see the same colours. Chroma follows the H.264/HEVC default sample location (type 0: co-sited with
 * even luma columns, half-way between luma rows): bilinear when enlarging, [1 2 1] x 2-row average when shrinking.
 * Decoding also takes 10-bit samples, 4:2:2 / 4:4:4 chroma and HDR (PQ / HLG), which is tone-mapped to SDR BT.709.
 */
object Yuv {

    enum class Standard(val kr: Double, val kb: Double) {
        BT601(0.299, 0.114), BT709(0.2126, 0.0722), BT2020(0.2627, 0.0593)
    }

    /** SDR: BT.709/601 gamma. PQ (SMPTE ST 2084) and HLG (ARIB STD-B67) are HDR. */
    enum class Transfer { SDR, PQ, HLG }

    data class Spec(val standard: Standard, val fullRange: Boolean, val transfer: Transfer = Transfer.SDR) {
        val isHdr get() = transfer != Transfer.SDR

        /** What [toRgb] produces and the encoder writes: HDR becomes SDR BT.709, limited range. */
        val sdr: Spec get() = if (isHdr) Spec(Standard.BT709, false) else this

        override fun toString() = "${standard.name} ${if (fullRange) "full" else "limited"} range" + if (isHdr) " HDR ${transfer.name}" else ""

        companion object {
            /** What players assume for untagged video: BT.709 from 720p up, BT.601 below; limited range. */
            fun guess(width: Int, height: Int) = Spec(if (minOf(width, height) >= 720) Standard.BT709 else Standard.BT601, false)
        }
    }

    /**
     * One plane: [data] from [offset], [rowStride] bytes per row, [pixelStride] bytes between samples.
     * [wide]: 16-bit little-endian samples with the value in the high bits (P010 and similar), else 8-bit.
     */
    class Plane(val data: ByteArray, val rowStride: Int, val pixelStride: Int, val offset: Int = 0, val wide: Boolean = false) {
        /** sample on the 8-bit scale (10-bit values keep their fraction) */
        fun at(x: Int, y: Int): Float {
            val i = offset + y * rowStride + x * pixelStride
            return if (wide) ((data[i].toInt() and 0xFF) or ((data[i + 1].toInt() and 0xFF) shl 8)) / 256f
            else (data[i].toInt() and 0xFF).toFloat()
        }
    }

    private class Coeffs(spec: Spec) {
        val kr = spec.standard.kr
        val kb = spec.standard.kb
        val kg = 1 - kr - kb
        val yOff = if (spec.fullRange) 0.0 else 16.0
        val yScale = if (spec.fullRange) 255.0 else 219.0
        val cScale = if (spec.fullRange) 255.0 else 224.0
    }

    /**
     * Decodes the [w]x[h] crop starting at ([cx], [cy]) to 8-bit RGB (SDR; HDR is tone-mapped, see [ToneMap]).
     * [subX]/[subY]: chroma subsampling, 2/2 = 4:2:0, 2/1 = 4:2:2, 1/1 = 4:4:4.
     * [rows]: --parallel_color, splits the rows over several threads (same result).
     */
    fun toRgb(yp: Plane, up: Plane, vp: Plane, cx: Int, cy: Int, w: Int, h: Int, spec: Spec, subX: Int = 2, subY: Int = 2, rows: RowPool? = null): Img {
        val k = Coeffs(spec)
        val tone = if (spec.isHdr) toneMap(spec) else null
        val out = Img(w, h, 3)
        val o = out.data
        val cw = (cx + w + subX - 1) / subX
        val ch = (cy + h + subY - 1) / subY
        val rv = (2 * (1 - k.kr) * 255.0 / k.cScale).toFloat()
        val bu = (2 * (1 - k.kb) * 255.0 / k.cScale).toFloat()
        val gu = (-2 * (1 - k.kb) * k.kb / k.kg * 255.0 / k.cScale).toFloat()
        val gv = (-2 * (1 - k.kr) * k.kr / k.kg * 255.0 / k.cScale).toFloat()
        val ys = (255.0 / k.yScale).toFloat()
        val yOff = k.yOff.toFloat()
        RowPool.run(rows, h) { yFrom, yTo ->
        val uRow = FloatArray(w)
        val vRow = FloatArray(w)
        for (y in yFrom until yTo) {
            val sy = cy + y
            var j0: Int
            var ay: Float
            if (subY == 2) {
                // chroma rows sit half-way between luma rows
                val fy = (sy - 0.5) / 2.0
                j0 = kotlin.math.floor(fy).toInt()
                ay = (fy - j0).toFloat()
                if (j0 < 0) { j0 = 0; ay = 0f }
                if (j0 >= ch - 1) { j0 = ch - 1; ay = 0f }
            } else { j0 = minOf(sy, ch - 1); ay = 0f }
            val j1 = minOf(j0 + 1, ch - 1)
            for (x in 0 until w) {
                val sx = cx + x
                // co-sited with even luma columns
                val i0 = minOf(sx / subX, cw - 1)
                val ax = if (subX == 2 && sx % 2 == 1 && i0 + 1 < cw) 0.5f else 0f
                val i1 = minOf(i0 + 1, cw - 1)
                fun lerp(p: Plane): Float {
                    val a = p.at(i0, j0) * (1 - ax) + p.at(i1, j0) * ax
                    if (ay == 0f) return a
                    val b = p.at(i0, j1) * (1 - ax) + p.at(i1, j1) * ax
                    return a * (1 - ay) + b * ay
                }
                uRow[x] = lerp(up) - 128f
                vRow[x] = lerp(vp) - 128f
            }
            var q = y * w * 3
            for (x in 0 until w) {
                val yy = (yp.at(cx + x, sy) - yOff) * ys
                val u = uRow[x]; val v = vRow[x]
                val r = yy + rv * v
                val g = yy + gu * u + gv * v
                val b = yy + bu * u
                if (tone != null) tone.map(r / 255f, g / 255f, b / 255f, o, q)
                else {
                    o[q] = clamp(r.roundToInt()).toByte()
                    o[q + 1] = clamp(g.roundToInt()).toByte()
                    o[q + 2] = clamp(b.roundToInt()).toByte()
                }
                q += 3
            }
        }
        }
        return out
    }

    @Volatile private var toneCache: Pair<Spec, ToneMap>? = null

    private fun toneMap(spec: Spec): ToneMap {
        toneCache?.let { if (it.first == spec) return it.second }
        return ToneMap(spec.transfer, spec.standard).also { toneCache = spec to it }
    }

    /**
     * HDR R'G'B' (0..1) -> SDR BT.709 8-bit, the way an SDR screen should show it:
     * PQ / HLG to light (HLG with the BT.2100 1000-nit display OOTF), 203 nits = SDR white (BT.2408),
     * BT.2020 -> BT.709 primaries, highlights above [KNEE] rolled off smoothly into the SDR range (applied to the
     * brightest channel so hues stay put), then BT.1886 (gamma 2.4) encoding.
     */
    class ToneMap(private val transfer: Transfer, standard: Standard, peakNits: Double = 1000.0) {
        private val eotf = FloatArray(LUT + 2)
        private val toLinear709 = standard == Standard.BT2020
        private val peak = (peakNits / REF_WHITE).toFloat()
        private val tm = (peak - KNEE) / (1 - KNEE)
        private val out = ByteArray(LUT + 1)

        init {
            for (i in 0..LUT + 1) {
                val v = minOf(i, LUT).toDouble() / LUT
                eotf[i] = when (transfer) {
                    Transfer.PQ -> (pq(v) / REF_WHITE).toFloat()
                    else -> hlgInverseOetf(v).toFloat() // scene light 0..1; OOTF needs the luminance, applied per pixel
                }
            }
            // indexed by sqrt(light), so the dark end (where the eye is most sensitive) gets most of the entries
            for (i in 0..LUT) {
                val s = i.toDouble() / LUT
                out[i] = (255 * s.pow(2 / 2.4)).roundToInt().coerceIn(0, 255).toByte()
            }
        }

        private fun lin(v: Float): Float {
            val x = (if (v < 0f) 0f else if (v > 1f) 1f else v) * LUT
            val i = x.toInt()
            val f = x - i
            return eotf[i] + (eotf[i + 1] - eotf[i]) * f
        }

        /** smooth roll-off: identity below KNEE, reaches 1 at the peak, slope 1 at the knee */
        private fun curve(x: Float): Float {
            if (x <= KNEE) return x
            if (x >= peak) return 1f
            val t = (x - KNEE) / (1 - KNEE)
            return KNEE + (1 - KNEE) * t * (1 + t / (tm * tm)) / (1 + t)
        }

        fun map(r: Float, g: Float, b: Float, o: ByteArray, q: Int) {
            var lr = lin(r); var lg = lin(g); var lb = lin(b)
            if (transfer == Transfer.HLG) {
                // BT.2100 OOTF for a 1000-nit display: Fd = 1000 * Ys^(1.2-1) * E
                val ys = 0.2627f * lr + 0.6780f * lg + 0.0593f * lb
                val s = if (ys > 0f) (1000.0 * ys.toDouble().pow(0.2) / REF_WHITE).toFloat() else 0f
                lr *= s; lg *= s; lb *= s
            }
            if (toLinear709) {
                // ITU-R BT.2087 (rows sum to 1, so greys stay grey)
                val r2 = 1.660491f * lr - 0.587641f * lg - 0.072850f * lb
                val g2 = -0.124550f * lr + 1.132900f * lg - 0.008350f * lb
                val b2 = -0.018151f * lr - 0.100579f * lg + 1.118730f * lb
                lr = maxOf(r2, 0f); lg = maxOf(g2, 0f); lb = maxOf(b2, 0f)
            }
            val m = maxOf(lr, lg, lb)
            if (m > KNEE) {
                val s = curve(m) / m
                lr *= s; lg *= s; lb *= s
            }
            o[q] = enc(lr); o[q + 1] = enc(lg); o[q + 2] = enc(lb)
        }

        private fun enc(l: Float): Byte = out[(sqrt(if (l > 1f) 1f else l) * LUT + 0.5f).toInt()]

        companion object {
            const val LUT = 4096
            /** SDR white in nits (ITU-R BT.2408 HDR reference white) */
            const val REF_WHITE = 203.0
            const val KNEE = 0.8f

            /** PQ signal 0..1 -> nits */
            fun pq(v: Double): Double {
                val m1 = 2610.0 / 16384; val m2 = 2523.0 / 4096 * 128
                val c1 = 3424.0 / 4096; val c2 = 2413.0 / 4096 * 32; val c3 = 2392.0 / 4096 * 32
                val p = v.pow(1 / m2)
                return 10000 * (maxOf(p - c1, 0.0) / (c2 - c3 * p)).pow(1 / m1)
            }

            /** HLG signal 0..1 -> scene light 0..1 */
            fun hlgInverseOetf(v: Double): Double {
                val a = 0.17883277; val b = 1 - 4 * a; val c = 0.5 - a * kotlin.math.ln(4 * a)
                return if (v <= 0.5) v * v / 3 else (exp((v - c) / a) + b) / 12
            }
        }
    }

    /** I420 planes (Y w*h, U and V (w/2)*(h/2)) for an even-sized RGB image (SDR; [spec] must not be HDR). */
    fun fromRgb(img: Img, spec: Spec, rows: RowPool? = null): Array<ByteArray> {
        require(!spec.isHdr) { "encoding is SDR only" }
        val k = Coeffs(spec)
        val w = img.w; val h = img.h
        val cw = w / 2; val chh = h / 2
        val yOut = ByteArray(w * h)
        val pb = FloatArray(w * h)
        val pr = FloatArray(w * h)
        val d = img.data
        RowPool.run(rows, h) { yFrom, yTo ->
        for (p in yFrom * w until yTo * w) {
            val r = (d[p * 3].toInt() and 0xFF) / 255.0
            val g = (d[p * 3 + 1].toInt() and 0xFF) / 255.0
            val b = (d[p * 3 + 2].toInt() and 0xFF) / 255.0
            val ey = k.kr * r + k.kg * g + k.kb * b
            yOut[p] = clamp((k.yOff + k.yScale * ey).roundToInt()).toByte()
            pb[p] = ((b - ey) / (2 * (1 - k.kb))).toFloat()
            pr[p] = ((r - ey) / (2 * (1 - k.kr))).toFloat()
        }
        }
        val u = ByteArray(cw * chh)
        val v = ByteArray(cw * chh)
        RowPool.run(rows, chh) { jFrom, jTo ->
        for (j in jFrom until jTo) for (i in 0 until cw) {
            var su = 0f; var sv = 0f
            for (row in 2 * j..2 * j + 1) {
                val base = row * w
                val xm = maxOf(2 * i - 1, 0); val xc = 2 * i; val xp = minOf(2 * i + 1, w - 1)
                su += pb[base + xm] + 2 * pb[base + xc] + pb[base + xp]
                sv += pr[base + xm] + 2 * pr[base + xc] + pr[base + xp]
            }
            u[j * cw + i] = clamp((128 + k.cScale * su / 8).roundToInt()).toByte()
            v[j * cw + i] = clamp((128 + k.cScale * sv / 8).roundToInt()).toByte()
        }
        }
        return arrayOf(yOut, u, v)
    }

    /** Rotates an image clockwise by 0/90/180/270 degrees. */
    fun rotate(img: Img, degrees: Int): Img {
        if (degrees % 360 == 0) return img
        val w = img.w; val h = img.h; val c = img.c
        val out = if (degrees % 180 == 0) Img(w, h, c) else Img(h, w, c)
        for (y in 0 until h) for (x in 0 until w) {
            val (nx, ny) = when (degrees) {
                90 -> (h - 1 - y) to x
                180 -> (w - 1 - x) to (h - 1 - y)
                else -> y to (w - 1 - x) // 270
            }
            val s = (y * w + x) * c
            val t = (ny * out.w + nx) * c
            for (ch in 0 until c) out.data[t + ch] = img.data[s + ch]
        }
        return out
    }

    private fun clamp(v: Int) = if (v < 0) 0 else if (v > 255) 255 else v
}
