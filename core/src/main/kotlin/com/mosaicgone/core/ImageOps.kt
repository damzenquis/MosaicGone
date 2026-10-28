// Modified for MosaicGone, 2026: translated to Kotlin from util/image_processing.py of DeepMosaics
// (https://github.com/HypoX64/DeepMosaics, GPL-3.0), with the OpenCV primitives it relies on written out and later
// changes for this app. See THIRD_PARTY_NOTICES.md and third_party_licenses/deepmosaics/LICENSE.
package com.mosaicgone.core

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** Ports of util/image_processing.py plus the OpenCV primitives it relies on. */
object ImageOps {

    enum class Interp(val taps: Int) { LINEAR(2), CUBIC(4), LANCZOS4(8) }

    // ------------------------------------------------------------------ resize

    fun resize(src: Img, nw: Int, nh: Int, interp: Interp = Interp.LINEAR): Img =
        resizeRegion(src, nw, nh, 0, 0, nw, nh, interp)

    /**
     * Equivalent to cv2.resize(src, (nw, nh))[ry:ry+rh, rx:rx+rw] but only computes the window,
     * so a small crop of a large up-scaled mask costs nothing extra.
     * Uses OpenCV's pixel-centre mapping and edge clamping.
     */
    fun resizeRegion(src: Img, nw: Int, nh: Int, rx: Int, ry: Int, rw: Int, rh: Int, interp: Interp = Interp.LINEAR): Img {
        require(nw > 0 && nh > 0 && rw > 0 && rh > 0) { "bad resize ${nw}x$nh" }
        if (nw == src.w && nh == src.h && rx == 0 && ry == 0 && rw == nw && rh == nh) return src.copy()
        val taps = interp.taps
        val xi = IntArray(rw * taps)
        val xw = FloatArray(rw * taps)
        val yi = IntArray(rh * taps)
        val yw = FloatArray(rh * taps)
        coeffs(src.w, nw, rx, rw, interp, xi, xw)
        coeffs(src.h, nh, ry, rh, interp, yi, yw)
        val c = src.c
        val out = Img(rw, rh, c)
        val s = src.data
        val o = out.data
        if (interp == Interp.LANCZOS4) {
            lanczosFixedPoint(src, xi, xw, yi, yw, rw, rh, out)
            return out
        }
        val tmp = FloatArray(c)
        for (y in 0 until rh) {
            for (x in 0 until rw) {
                java.util.Arrays.fill(tmp, 0f)
                for (ty in 0 until taps) {
                    val wy = yw[y * taps + ty]
                    if (wy == 0f) continue
                    val row = yi[y * taps + ty] * src.w
                    for (tx in 0 until taps) {
                        val wgt = wy * xw[x * taps + tx]
                        if (wgt == 0f) continue
                        val p = (row + xi[x * taps + tx]) * c
                        for (ch in 0 until c) tmp[ch] += wgt * (s[p + ch].toInt() and 0xFF)
                    }
                }
                val q = (y * rw + x) * c
                for (ch in 0 until c) o[q + ch] = clamp255(Math.round(tmp[ch])).toByte()
            }
        }
        return out
    }

    /**
     * OpenCV's 8-bit path for INTER_LANCZOS4: weights quantised to shorts (x2048), integer sums,
     * result = (sum + 2^21) >> 22 saturated. Reproduces cv2.resize exactly.
     */
    private fun lanczosFixedPoint(src: Img, xi: IntArray, xw: FloatArray, yi: IntArray, yw: FloatArray, rw: Int, rh: Int, out: Img) {
        val c = src.c
        val s = src.data
        val o = out.data
        val ax = IntArray(xw.size) { Math.round(xw[it] * 2048f) }
        val ay = IntArray(yw.size) { Math.round(yw[it] * 2048f) }
        val acc = LongArray(c)
        for (y in 0 until rh) {
            for (x in 0 until rw) {
                java.util.Arrays.fill(acc, 0L)
                for (ty in 0 until 8) {
                    val by = ay[y * 8 + ty]
                    if (by == 0) continue
                    val row = yi[y * 8 + ty] * src.w
                    for (ch in 0 until c) {
                        var h = 0L
                        for (tx in 0 until 8) h += ax[x * 8 + tx].toLong() * (s[(row + xi[x * 8 + tx]) * c + ch].toInt() and 0xFF)
                        acc[ch] += by * h
                    }
                }
                val q = (y * rw + x) * c
                for (ch in 0 until c) o[q + ch] = clamp255(((acc[ch] + (1L shl 21)) shr 22).toInt()).toByte()
            }
        }
    }

    private fun coeffs(srcLen: Int, dstLen: Int, start: Int, count: Int, interp: Interp, idx: IntArray, wgt: FloatArray) {
        val scale = srcLen.toDouble() / dstLen
        for (k in 0 until count) {
            val d = start + k
            val f = (d + 0.5) * scale - 0.5
            var i0 = floor(f).toInt()
            var a = (f - i0).toFloat()
            if (interp == Interp.LINEAR) {
                if (i0 < 0) { i0 = 0; a = 0f }
                if (i0 >= srcLen - 1) { i0 = srcLen - 1; a = 0f }
                idx[k * 2] = i0
                idx[k * 2 + 1] = min(i0 + 1, srcLen - 1)
                wgt[k * 2] = 1f - a
                wgt[k * 2 + 1] = a
            } else if (interp == Interp.LANCZOS4) {
                // cv2 interpolateLanczos4: 8 taps i0-3..i0+4, sinc(d)*sinc(d/4), normalised
                var sum = 0f
                for (t in 0 until 8) {
                    val d = (a + 3 - t).toDouble()
                    val w = if (kotlin.math.abs(d) < 1e-7) 1.0 else {
                        val y = Math.PI * d
                        4.0 * kotlin.math.sin(y) * kotlin.math.sin(y / 4) / (y * y)
                    }
                    idx[k * 8 + t] = (i0 - 3 + t).coerceIn(0, srcLen - 1)
                    wgt[k * 8 + t] = w.toFloat()
                    sum += w.toFloat()
                }
                for (t in 0 until 8) wgt[k * 8 + t] /= sum
            } else {
                val A = -0.75f
                val w0 = ((A * (a + 1) - 5 * A) * (a + 1) + 8 * A) * (a + 1) - 4 * A
                val w1 = ((A + 2) * a - (A + 3)) * a * a + 1
                val w2 = ((A + 2) * (1 - a) - (A + 3)) * (1 - a) * (1 - a) + 1
                val w3 = 1f - w0 - w1 - w2
                val ws = floatArrayOf(w0, w1, w2, w3)
                for (t in 0 until 4) {
                    idx[k * 4 + t] = (i0 - 1 + t).coerceIn(0, srcLen - 1)
                    wgt[k * 4 + t] = ws[t]
                }
            }
        }
    }

    /**
     * cv2.resize(..., INTER_AREA) for shrinking: every output pixel is the coverage-weighted average of the
     * source pixels under it, so detail is averaged instead of skipped (no aliasing / moire). Upscaling falls back to linear.
     */
    fun resizeArea(src: Img, nw: Int, nh: Int): Img {
        if (nw >= src.w && nh >= src.h) return resize(src, nw, nh)
        val (xi, xw, xn) = areaTable(src.w, nw)
        val (yi, yw, yn) = areaTable(src.h, nh)
        val c = src.c
        val s = src.data
        val out = Img(nw, nh, c)
        val row = FloatArray(src.w * c)
        val acc = FloatArray(nw * c)
        for (y in 0 until nh) {
            java.util.Arrays.fill(acc, 0f)
            for (k in yn[y] until yn[y + 1]) {
                val wy = yw[k]
                val base = yi[k] * src.w * c
                for (p in 0 until src.w * c) row[p] = (s[base + p].toInt() and 0xFF).toFloat()
                for (x in 0 until nw) for (t in xn[x] until xn[x + 1]) {
                    val wgt = wy * xw[t]
                    val sp = xi[t] * c
                    for (ch in 0 until c) acc[x * c + ch] += wgt * row[sp + ch]
                }
            }
            // cvRound: round half to even
            for (p in 0 until nw * c) out.data[y * nw * c + p] = clamp255(Math.rint(acc[p].toDouble()).toInt()).toByte()
        }
        return out
    }

    /** For each destination pixel: source indices and weights (overlap / scale); index ranges via the offsets array. */
    private fun areaTable(srcLen: Int, dstLen: Int): Triple<IntArray, FloatArray, IntArray> {
        if (dstLen >= srcLen) {
            return Triple(IntArray(dstLen) { it * srcLen / dstLen }, FloatArray(dstLen) { 1f }, IntArray(dstLen + 1) { it })
        }
        val scale = srcLen.toDouble() / dstLen
        val idx = ArrayList<Int>()
        val w = ArrayList<Float>()
        val off = IntArray(dstLen + 1)
        for (d in 0 until dstLen) {
            off[d] = idx.size
            val a = d * scale
            val b = minOf((d + 1) * scale, srcLen.toDouble())
            var sx = floor(a).toInt()
            while (sx < b) {
                val cover = minOf(b, sx + 1.0) - maxOf(a, sx.toDouble())
                if (cover > 1e-9) { idx.add(sx); w.add((cover / scale).toFloat()) }
                sx++
            }
        }
        off[dstLen] = idx.size
        return Triple(idx.toIntArray(), w.toFloatArray(), off)
    }

    /** impro.resize: scale so the short side equals [size], keeping aspect ratio. */
    fun resizeShort(img: Img, size: Int, interp: Interp = Interp.LINEAR): Img {
        if (min(img.w, img.h) == size) return img
        return if (img.w >= img.h) resize(img, max(1, (size.toDouble() * img.w / img.h).toInt()), size, interp)
        else resize(img, size, max(1, (size.toDouble() * img.h / img.w).toInt()), interp)
    }

    /** Pads [p] pixels on every side by mirroring (torch 'reflect' = cv2 BORDER_REFLECT_101). */
    fun reflectPad(img: Img, p: Int): Img {
        if (p <= 0) return img
        val out = Img(img.w + 2 * p, img.h + 2 * p, img.c)
        for (y in 0 until out.h) {
            val sy = reflect101(y - p, img.h)
            for (x in 0 until out.w) {
                val sx = reflect101(x - p, img.w)
                for (ch in 0 until img.c) out[x, y, ch] = img[sx, sy, ch]
            }
        }
        return out
    }

    // ------------------------------------------------------------------ blur

    private fun reflect101(i: Int, n: Int): Int {
        if (n == 1) return 0
        var p = i
        while (p < 0 || p >= n) p = if (p < 0) -p else 2 * n - 2 - p
        return p
    }

    /** cv2.blur(src, (kw, kh)) — normalized box filter, BORDER_REFLECT_101, O(1) per pixel. */
    fun blur(src: Img, kw: Int, kh: Int = kw): Img {
        val kx = max(1, kw)
        val ky = max(1, kh)
        val w = src.w
        val h = src.h
        val c = src.c
        if (kx == 1 && ky == 1) return src.copy()
        val ax = kx / 2
        val ay = ky / 2
        val s = src.data
        val col = IntArray(w * c)
        fun addRow(r: Int, sign: Int) {
            val base = r * w * c
            for (k in 0 until w * c) col[k] += sign * (s[base + k].toInt() and 0xFF)
        }
        for (t in 0 until ky) addRow(reflect101(t - ay, h), 1)
        val xIdx = IntArray(w + kx) { reflect101(it - ax, w) }
        val area = kx * ky
        val half = area / 2
        val out = Img(w, h, c)
        val o = out.data
        for (y in 0 until h) {
            for (ch in 0 until c) {
                var sum = 0
                for (t in 0 until kx) sum += col[xIdx[t] * c + ch]
                var q = y * w * c + ch
                for (x in 0 until w) {
                    o[q] = ((sum + half) / area).toByte()
                    sum += col[xIdx[x + kx] * c + ch] - col[xIdx[x] * c + ch]
                    q += c
                }
            }
            if (y + 1 < h) {
                addRow(reflect101(y + 1 - ay + ky - 1, h), 1)
                addRow(reflect101(y - ay, h), -1)
            }
        }
        return out
    }

    // ------------------------------------------------------------------ masks

    /** cv2.threshold(mask, t, 255, THRESH_BINARY): v > t -> 255. */
    fun threshold(mask: Img, t: Double): Img {
        val out = Img(mask.w, mask.h, mask.c)
        for (i in mask.data.indices) if ((mask.data[i].toInt() and 0xFF) > t) out.data[i] = -1
        return out
    }

    /** impro.mask_threshold */
    fun maskThreshold(mask: Img, exMun: Int, threshold: Int): Img {
        var m = threshold(mask, threshold.toDouble())
        m = blur(m, max(1, exMun))
        return threshold(m, threshold / 5.0)
    }

    /** 8-connected components of non-zero pixels. Returns (labels, sizes) with label 0 = background. */
    private fun label(mask: Img): Pair<IntArray, IntArray> {
        val w = mask.w
        val h = mask.h
        val labels = IntArray(w * h)
        val sizes = ArrayList<Int>().apply { add(0) }
        val stack = IntArray(w * h)
        val d = mask.data
        for (start in 0 until w * h) {
            if (d[start].toInt() == 0 || labels[start] != 0) continue
            val id = sizes.size
            var sp = 0
            var count = 0
            stack[sp++] = start
            labels[start] = id
            while (sp > 0) {
                val p = stack[--sp]
                count++
                val px = p % w
                val py = p / w
                for (dy in -1..1) {
                    val ny = py + dy
                    if (ny < 0 || ny >= h) continue
                    for (dx in -1..1) {
                        val nx = px + dx
                        if (nx < 0 || nx >= w) continue
                        val q = ny * w + nx
                        if (d[q].toInt() != 0 && labels[q] == 0) {
                            labels[q] = id
                            stack[sp++] = q
                        }
                    }
                }
            }
            sizes.add(count)
        }
        return labels to sizes.toIntArray()
    }

    /** Area (pixel count) of the largest blob after thresholding at 127 — stands in for impro.mask_area. */
    fun maskArea(mask: Img): Int {
        val (_, sizes) = label(threshold(mask, 127.0))
        return sizes.maxOrNull() ?: 0
    }

    /** impro.find_mostlikely_ROI: keep only the largest blob, filled (like cv2.fillPoly of its outer contour). */
    fun findMostLikelyROI(mask: Img): Img {
        val (labels, sizes) = label(mask)
        if (sizes.size <= 1) return mask.copy()
        var best = 1
        for (i in 1 until sizes.size) if (sizes[i] > sizes[best]) best = i
        val w = mask.w
        val h = mask.h
        var x0 = w; var y0 = h; var x1 = -1; var y1 = -1
        for (p in labels.indices) if (labels[p] == best) {
            val x = p % w; val y = p / w
            if (x < x0) x0 = x; if (x > x1) x1 = x
            if (y < y0) y0 = y; if (y > y1) y1 = y
        }
        // flood the outside of the blob inside its bbox (padded by 1); anything not reached is inside
        val bw = x1 - x0 + 3
        val bh = y1 - y0 + 3
        val outside = BooleanArray(bw * bh)
        val stack = IntArray(bw * bh)
        var sp = 0
        fun isBlob(gx: Int, gy: Int): Boolean {
            val x = gx + x0 - 1; val y = gy + y0 - 1
            return x in 0 until w && y in 0 until h && labels[y * w + x] == best
        }
        outside[0] = true
        stack[sp++] = 0
        while (sp > 0) {
            val p = stack[--sp]
            val gx = p % bw; val gy = p / bw
            val n4 = intArrayOf(gx - 1, gy, gx + 1, gy, gx, gy - 1, gx, gy + 1)
            for (k in 0 until 4) {
                val nx = n4[k * 2]; val ny = n4[k * 2 + 1]
                if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue
                val q = ny * bw + nx
                if (!outside[q] && !isBlob(nx, ny)) { outside[q] = true; stack[sp++] = q }
            }
        }
        val out = Img(w, h, 1)
        for (gy in 1 until bh - 1) for (gx in 1 until bw - 1) {
            if (!outside[gy * bw + gx]) out.data[(gy + y0 - 1) * w + (gx + x0 - 1)] = -1
        }
        return out
    }

    /** cv2.boundingRect of the non-zero pixels: (x, y, w, h), all 0 when empty. */
    fun boundingRect(mask: Img): IntArray {
        var x0 = mask.w; var y0 = mask.h; var x1 = -1; var y1 = -1
        for (y in 0 until mask.h) {
            val row = y * mask.w
            for (x in 0 until mask.w) if (mask.data[row + x].toInt() != 0) {
                if (x < x0) x0 = x; if (x > x1) x1 = x
                if (y < y0) y0 = y; if (y > y1) y1 = y
            }
        }
        return if (x1 < 0) intArrayOf(0, 0, 0, 0) else intArrayOf(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    }

    /** impro.boundingSquare -> (cx, cy, halfsize, area) */
    fun boundingSquare(mask: Img, exMul: Double): IntArray {
        val area = maskArea(mask)
        if (area == 0) return intArrayOf(0, 0, 0, 0)
        val (x, y, bw, bh) = boundingRect(mask).let { listOf(it[0], it[1], it[2], it[3]) }
        var cx = (x + bw / 2.0).toInt()
        var cy = (y + bh / 2.0).toInt()
        var size: Int
        val halfsize: Int
        val h = mask.h
        val w = mask.w
        val s0 = max(bw, bh)
        if (s0 * exMul > min(h, w)) {
            size = min(h, w)
            halfsize = min(h, w) / 2
        } else {
            halfsize = (exMul * s0 / 2.0).toInt()
            size = halfsize * 2
        }
        var p0x = cx - halfsize; var p0y = cy - halfsize
        var p1x = cx + halfsize; var p1y = cy + halfsize
        if (p0x < 0) { p0x = 0; p1x = size }
        if (p0y < 0) { p0y = 0; p1y = size }
        if (p1x > w) { p1x = w; p0x = w - size }
        if (p1y > h) { p1y = h; p0y = h - size }
        cx = floor((p0x + p1x) / 2.0).toInt()
        cy = floor((p0y + p1y) / 2.0).toInt()
        return intArrayOf(cx, cy, halfsize, area)
    }

    // ------------------------------------------------------------------ compositing

    /** impro.replace_mosaic — modifies and returns [origin]. */
    fun replaceMosaic(origin: Img, fake: Img, mask: Img, x: Int, y: Int, size: Int, noFeather: Boolean): Img {
        val fakeR = resize(fake, size * 2, size * 2, Interp.CUBIC)
        if (noFeather) {
            origin.paste(fakeR, x - size, y - size)
            return origin
        }
        val eclosion = size / 10 + 2
        var m = resizeRegion(mask, origin.w, origin.h, x - size, y - size, size * 2, size * 2)
        m = blur(m, eclosion)
        val n = size * 2
        for (yy in 0 until n) for (xx in 0 until n) {
            val a = (m.data[yy * n + xx].toInt() and 0xFF) / 255f
            if (a == 0f) continue
            val ox = x - size + xx
            val oy = y - size + yy
            for (ch in 0 until 3) {
                val v = origin[ox, oy, ch] * (1 - a) + fakeR[xx, yy, ch] * a
                origin[ox, oy, ch] = clamp255(v.toInt())
            }
        }
        return origin
    }

    internal fun clamp255(v: Int) = if (v < 0) 0 else if (v > 255) 255 else v
}
