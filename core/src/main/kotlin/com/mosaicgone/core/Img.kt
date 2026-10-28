package com.mosaicgone.core

/**
 * Interleaved 8-bit image. c = 3 is RGB, c = 1 is a gray mask.
 * (The Python code uses OpenCV BGR; here everything is RGB and the
 * channel order is swapped only where a network expects BGR.)
 */
class Img(val w: Int, val h: Int, val c: Int, val data: ByteArray = ByteArray(w * h * c)) {
    init {
        require(w > 0 && h > 0 && (c == 1 || c == 3)) { "bad image shape ${w}x${h}x$c" }
        require(data.size == w * h * c) { "data size ${data.size} != ${w * h * c}" }
    }

    operator fun get(x: Int, y: Int, ch: Int = 0): Int = data[(y * w + x) * c + ch].toInt() and 0xFF

    operator fun set(x: Int, y: Int, ch: Int, v: Int) {
        data[(y * w + x) * c + ch] = v.toByte()
    }

    fun copy() = Img(w, h, c, data.copyOf())

    /** Copy of the rectangle [x0, x0+cw) x [y0, y0+ch). Must be inside the image. */
    fun crop(x0: Int, y0: Int, cw: Int, ch: Int): Img {
        require(x0 >= 0 && y0 >= 0 && x0 + cw <= w && y0 + ch <= h) { "crop out of bounds" }
        val out = Img(cw, ch, c)
        val rowBytes = cw * c
        for (y in 0 until ch) {
            System.arraycopy(data, ((y0 + y) * w + x0) * c, out.data, y * rowBytes, rowBytes)
        }
        return out
    }

    fun paste(src: Img, x0: Int, y0: Int) {
        require(src.c == c && x0 >= 0 && y0 >= 0 && x0 + src.w <= w && y0 + src.h <= h) { "paste out of bounds" }
        val rowBytes = src.w * c
        for (y in 0 until src.h) {
            System.arraycopy(src.data, y * rowBytes, data, ((y0 + y) * w + x0) * c, rowBytes)
        }
    }
}
