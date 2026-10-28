package com.mosaicgone.core

import java.io.ByteArrayOutputStream

/**
 * Animated GIF reading and writing (GIF87a/89a), so GIFs are cleaned frame by frame and saved as GIF again
 * with the source's size, frame timing, loop count and colour palettes.
 */
object Gif {

    /**
     * A fully composited frame as shown to the viewer. Kept deflated (see [PackedImages]) so long GIFs fit in memory;
     * [image] and [opaque] unpack a fresh copy on each call.
     */
    class Frame(
        image: Img,
        opaque: BooleanArray,
        /** RGB palette the source used for this frame (local table, else global) */
        val palette: IntArray,
        val transparentIndex: Int,
        /** delay in 1/100 s */
        val delayCs: Int,
    ) {
        private val w = image.w
        private val h = image.h
        private val packed = PackedImages.pack(image.data)
        /** null = fully opaque */
        private val packedOpaque = if (opaque.all { it }) null else PackedImages.pack(ByteArray(opaque.size) { if (opaque[it]) 1 else 0 })

        val image: Img get() = Img(w, h, 3).also { PackedImages.unpack(packed, it.data) }

        /** false where the canvas is transparent (nothing drawn yet / cleared) */
        val opaque: BooleanArray
            get() {
                val z = packedOpaque ?: return BooleanArray(w * h) { true }
                val b = ByteArray(w * h).also { PackedImages.unpack(z, it) }
                return BooleanArray(b.size) { b[it].toInt() != 0 }
            }

        val hasTransparency get() = packedOpaque != null
    }

    class Animation(val width: Int, val height: Int, val frames: List<Frame>, val loopCount: Int) {
        val hasTransparency get() = frames.any { it.hasTransparency }
    }

    class FormatError(msg: String) : Exception(msg)

    // ------------------------------------------------------------------ decode

    private class Reader(val b: ByteArray) {
        var p = 0
        fun u8(): Int { if (p >= b.size) throw FormatError("GIF ends early"); return b[p++].toInt() and 0xFF }
        fun u16() = u8() or (u8() shl 8)
        fun table(n: Int) = IntArray(n) { (u8() shl 16) or (u8() shl 8) or u8() }
        fun subBlocks(): ByteArray {
            val out = ByteArrayOutputStream()
            while (true) {
                val n = u8()
                if (n == 0) break
                if (p + n > b.size) throw FormatError("GIF ends early")
                out.write(b, p, n); p += n
            }
            return out.toByteArray()
        }
    }

    fun decode(bytes: ByteArray): Animation {
        val r = Reader(bytes)
        val sig = String(bytes, 0, minOf(6, bytes.size), Charsets.US_ASCII)
        if (sig != "GIF87a" && sig != "GIF89a") throw FormatError("not a GIF")
        r.p = 6
        val w = r.u16(); val h = r.u16()
        val packed = r.u8(); r.u8(); r.u8() // background index, aspect
        val global = if (packed and 0x80 != 0) r.table(2 shl (packed and 7)) else null
        val canvas = IntArray(w * h) // ARGB, 0 = transparent
        var saved: IntArray? = null
        var disposal = 0; var delay = 0; var trans = -1; var loop = -1
        val frames = ArrayList<Frame>()
        loop@ while (r.p < bytes.size) {
            when (r.u8()) {
                0x21 -> {
                    val label = r.u8()
                    val data = r.subBlocks()
                    if (label == 0xF9 && data.size >= 4) {
                        val pk = data[0].toInt() and 0xFF
                        disposal = (pk shr 2) and 7
                        delay = (data[1].toInt() and 0xFF) or ((data[2].toInt() and 0xFF) shl 8)
                        trans = if (pk and 1 != 0) data[3].toInt() and 0xFF else -1
                    } else if (label == 0xFF && data.size >= 14 && String(data, 0, 11, Charsets.US_ASCII) == "NETSCAPE2.0" && data[11].toInt() == 1) {
                        loop = (data[12].toInt() and 0xFF) or ((data[13].toInt() and 0xFF) shl 8)
                    }
                }
                0x2C -> {
                    val fx = r.u16(); val fy = r.u16(); val fw = r.u16(); val fh = r.u16()
                    val pk = r.u8()
                    val local = if (pk and 0x80 != 0) r.table(2 shl (pk and 7)) else null
                    val pal = local ?: global ?: throw FormatError("GIF frame without a colour table")
                    val minCode = r.u8()
                    val idx = lzwDecode(r.subBlocks(), minCode, fw * fh)
                    if (disposal == 3) saved = canvas.copyOf()
                    val rows = if (pk and 0x40 != 0) interlacedRows(fh) else IntArray(fh) { it }
                    for (row in 0 until fh) {
                        val y = fy + rows[row]
                        if (y >= h) continue
                        for (x in 0 until fw) {
                            val cx = fx + x
                            if (cx >= w) continue
                            val ci = idx[row * fw + x].toInt() and 0xFF
                            if (ci == trans || ci >= pal.size) continue
                            canvas[y * w + cx] = (0xFF shl 24) or pal[ci]
                        }
                    }
                    val img = Img(w, h, 3)
                    val opaque = BooleanArray(w * h)
                    for (p in 0 until w * h) {
                        val c = canvas[p]
                        if (c ushr 24 != 0) {
                            opaque[p] = true
                            img.data[p * 3] = (c shr 16).toByte(); img.data[p * 3 + 1] = (c shr 8).toByte(); img.data[p * 3 + 2] = c.toByte()
                        }
                    }
                    frames.add(Frame(img, opaque, pal, trans, delay))
                    when (disposal) {
                        2 -> for (y in fy until minOf(h, fy + fh)) for (x in fx until minOf(w, fx + fw)) canvas[y * w + x] = 0
                        3 -> saved?.copyInto(canvas)
                    }
                    disposal = 0; delay = 0; trans = -1
                }
                0x3B -> break@loop
                0x00 -> {}
                else -> break@loop // tolerate trailing junk like browsers do
            }
        }
        if (frames.isEmpty()) throw FormatError("GIF has no frames")
        return Animation(w, h, frames, loop)
    }

    private fun interlacedRows(h: Int): IntArray {
        val order = ArrayList<Int>(h)
        for ((start, step) in listOf(0 to 8, 4 to 8, 2 to 4, 1 to 2)) { var y = start; while (y < h) { order.add(y); y += step } }
        return order.toIntArray()
    }

    private fun lzwDecode(data: ByteArray, minCode: Int, n: Int): ByteArray {
        val out = ByteArray(n)
        val clear = 1 shl minCode
        val eoi = clear + 1
        val prefix = IntArray(4096); val suffix = ByteArray(4096)
        for (i in 0 until clear) suffix[i] = i.toByte()
        val stack = ByteArray(4097)
        var codeSize = minCode + 1; var next = clear + 2; var old = -1
        var bits = 0; var acc = 0; var o = 0
        var i = 0
        while (o < n) {
            while (bits < codeSize) {
                if (i >= data.size) return out
                acc = acc or ((data[i++].toInt() and 0xFF) shl bits); bits += 8
            }
            val code = acc and ((1 shl codeSize) - 1)
            acc = acc ushr codeSize; bits -= codeSize
            if (code == clear) { codeSize = minCode + 1; next = clear + 2; old = -1; continue }
            if (code == eoi) break
            var sp = 0
            var cur = code
            if (old == -1) {
                if (code >= clear) return out
                out[o++] = code.toByte(); old = code; continue
            }
            val firstByte: Byte
            if (code < next) {
                while (cur >= clear) { stack[sp++] = suffix[cur]; cur = prefix[cur] }
                stack[sp++] = suffix[cur]
                firstByte = suffix[cur]
            } else { // KwKwK
                cur = old
                while (cur >= clear) { stack[sp++] = suffix[cur]; cur = prefix[cur] }
                stack[sp++] = suffix[cur]
                firstByte = suffix[cur]
                // the new code's string = old string + its first byte, written below
                for (k in sp downTo 1) stack[k] = stack[k - 1]
                stack[0] = firstByte; sp++
            }
            while (sp > 0 && o < n) out[o++] = stack[--sp]
            if (next < 4096) { prefix[next] = old; suffix[next] = firstByte; next++ }
            if (next == (1 shl codeSize) && codeSize < 12) codeSize++
            old = code
        }
        return out
    }

    // ------------------------------------------------------------------ encode

    /**
     * Writes [images] (one per source frame) as an animated GIF with the source's palettes, delays and loop count.
     * Opaque animations store only what changed since the previous frame (unchanged pixels transparent), like the
     * frame optimisation most GIFs already use; animations with transparency are stored as full frames.
     */
    fun encode(src: Animation, images: List<Img>): ByteArray {
        require(images.size == src.frames.size) { "frame count changed" }
        val w = src.width; val h = src.height
        val out = ByteArrayOutputStream()
        out.write("GIF89a".toByteArray(Charsets.US_ASCII))
        le16(out, w); le16(out, h)
        out.write(0); out.write(0); out.write(0) // no global table, background 0, aspect 0
        if (src.loopCount >= 0) {
            out.write(byteArrayOf(0x21, 0xFF.toByte(), 11)); out.write("NETSCAPE2.0".toByteArray(Charsets.US_ASCII))
            out.write(byteArrayOf(3, 1)); le16(out, src.loopCount); out.write(0)
        }
        val transparentCanvas = src.hasTransparency
        val shown = IntArray(w * h) { -1 } // what the viewer currently sees (RGB), -1 = nothing
        for ((k, f) in src.frames.withIndex()) {
            val img = images[k]
            require(img.w == w && img.h == h) { "frame size changed" }
            val opaque = if (transparentCanvas) f.opaque else null
            // palette: the source's; reserve a transparent slot (reuse the source's one if it had it)
            var pal = f.palette.copyOf()
            var trans = f.transparentIndex
            if (trans < 0 && pal.size < 256) { pal = pal.copyOf(pal.size + 1); trans = pal.size - 1 }
            val delta = !transparentCanvas && k > 0 && trans >= 0
            val mapper = Mapper(pal, trans)
            val idx = ByteArray(w * h)
            for (p in 0 until w * h) {
                if (opaque != null && !opaque[p] && trans >= 0) { idx[p] = trans.toByte(); shown[p] = -1; continue }
                val rgb = ((img.data[p * 3].toInt() and 0xFF) shl 16) or ((img.data[p * 3 + 1].toInt() and 0xFF) shl 8) or (img.data[p * 3 + 2].toInt() and 0xFF)
                val ci = mapper.map(rgb)
                val col = pal[ci]
                if (delta && shown[p] == col) { idx[p] = trans.toByte(); continue }
                idx[p] = ci.toByte(); shown[p] = col
            }
            // graphic control: disposal 2 (clear) for transparent animations, 1 (keep) otherwise
            val disposal = if (transparentCanvas) 2 else 1
            out.write(byteArrayOf(0x21, 0xF9.toByte(), 4))
            out.write((disposal shl 2) or (if (trans >= 0) 1 else 0)); le16(out, f.delayCs); out.write(if (trans >= 0) trans else 0); out.write(0)
            // image descriptor with a local colour table
            var bitsN = 1
            while ((1 shl bitsN) < pal.size) bitsN++
            out.write(0x2C); le16(out, 0); le16(out, 0); le16(out, w); le16(out, h)
            out.write(0x80 or (bitsN - 1))
            for (i in 0 until (1 shl bitsN)) {
                val c = if (i < pal.size) pal[i] else 0
                out.write(c shr 16 and 0xFF); out.write(c shr 8 and 0xFF); out.write(c and 0xFF)
            }
            val minCode = maxOf(2, bitsN)
            out.write(minCode)
            val lzw = lzwEncode(idx, minCode)
            var p = 0
            while (p < lzw.size) { val n = minOf(255, lzw.size - p); out.write(n); out.write(lzw, p, n); p += n }
            out.write(0)
        }
        out.write(0x3B)
        return out.toByteArray()
    }

    private fun le16(o: ByteArrayOutputStream, v: Int) { o.write(v and 0xFF); o.write(v shr 8 and 0xFF) }

    /** Nearest palette colour (exact matches first), cached. */
    private class Mapper(val pal: IntArray, val skip: Int) {
        private val cache = HashMap<Int, Int>()
        init { for (i in pal.indices) if (i != skip) cache.putIfAbsent(pal[i], i) }
        fun map(rgb: Int): Int = cache.getOrPut(rgb) {
            val r = rgb shr 16 and 0xFF; val g = rgb shr 8 and 0xFF; val b = rgb and 0xFF
            var best = 0; var bd = Long.MAX_VALUE
            for (i in pal.indices) {
                if (i == skip) continue
                val c = pal[i]
                val dr = r - (c shr 16 and 0xFF); val dg = g - (c shr 8 and 0xFF); val db = b - (c and 0xFF)
                // weighted (eyes are most sensitive to green)
                val d = 2L * dr * dr + 4L * dg * dg + 3L * db * db
                if (d < bd) { bd = d; best = i }
            }
            best
        }
    }

    private fun lzwEncode(idx: ByteArray, minCode: Int): ByteArray {
        val out = ByteArrayOutputStream()
        val clear = 1 shl minCode; val eoi = clear + 1
        var codeSize = minCode + 1; var next = clear + 2
        val dict = HashMap<Int, Int>() // (prefix shl 8) or byte -> code
        var acc = 0L; var bits = 0
        fun emit(code: Int) {
            acc = acc or (code.toLong() shl bits); bits += codeSize
            while (bits >= 8) { out.write((acc and 0xFF).toInt()); acc = acc ushr 8; bits -= 8 }
        }
        emit(clear)
        if (idx.isEmpty()) { emit(eoi); if (bits > 0) out.write((acc and 0xFF).toInt()); return out.toByteArray() }
        var cur = idx[0].toInt() and 0xFF
        for (i in 1 until idx.size) {
            val c = idx[i].toInt() and 0xFF
            val key = (cur shl 8) or c
            val found = dict[key]
            if (found != null) { cur = found; continue }
            emit(cur)
            if (next < 4096) {
                dict[key] = next++
                if (next > (1 shl codeSize) && codeSize < 12) codeSize++
            } else {
                emit(clear); dict.clear(); codeSize = minCode + 1; next = clear + 2
            }
            cur = c
        }
        emit(cur)
        emit(eoi)
        if (bits > 0) out.write((acc and 0xFF).toInt())
        return out.toByteArray()
    }
}
