package com.mosaicgone.core

/**
 * MosaicGone's video restorer (slim BasicVSR++, model-development/training/dmtrain/basicvsrpp.py, exported by
 * model-development/training/export_mosaicgone.py):
 * [T] square crops around the mosaic, resized to 256 x 256, in; the same [T] restored crops out. It looks at all of
 * them at once (forwards and backwards in time), so a stretch of video is cleaned in overlapping chunks of [T].
 * [short] is the 4-frame export, for photos (one crop repeated) and stretches of up to 4 frames, loaded on first use
 * (most videos never need it: less memory); [net] (16 frames) may be null for photos.
 * Thread-confined like [Net]: used from the AI thread.
 */
class Restorer(private val net: Net?, shortNet: Lazy<Net>) {
    constructor(net: Net?, short: Net) : this(net, lazyOf(short))

    private val short by shortNet


    companion object {
        const val T = 16
        const val T_SHORT = 4
        /** frames shared by neighbouring chunks, cross-faded so chunk joins do not jump */
        const val OVERLAP = 4
        private const val N = Nets.CROP * Nets.CROP

        /** A crop of any size -> 256 x 256 as the training crops were made: area-averaged when shrinking, cubic when growing. */
        fun toCrop(img: Img): Img =
            if (img.w > Nets.CROP) ImageOps.resizeArea(img, Nets.CROP, Nets.CROP)
            else ImageOps.resize(img, Nets.CROP, Nets.CROP, ImageOps.Interp.CUBIC)

        /** (1, len, 3, 256, 256) input from 256 x 256 crops; missing slots repeat the last crop. */
        fun input(crops: List<Img>, len: Int): Net.Tensor {
            require(crops.isNotEmpty() && crops.size <= len) { "restorer chunk of ${crops.size} crops for $len slots" }
            val d = FloatArray(len * 3 * N)
            for (i in 0 until len) Nets.fillRestorer(crops[minOf(i, crops.size - 1)], d, i)
            return Net.Tensor(d, longArrayOf(1, len.toLong(), 3, Nets.CROP.toLong(), Nets.CROP.toLong()))
        }

        /** Adds [w] x frame [i] of a restorer output (RGB planes 0..1) to [acc] (RGB planes, 0..255 scale). */
        fun accumulate(out: Net.Tensor, i: Int, w: Float, acc: FloatArray) {
            val base = i * 3 * N
            for (k in 0 until 3 * N) acc[k] += w * out.data[base + k] * 255f
        }

        /** [acc] / [wsum] (RGB planes) -> 256 x 256 RGB image */
        fun toImg(acc: FloatArray, wsum: Float): Img {
            val img = Img(Nets.CROP, Nets.CROP, 3)
            val inv = 1f / wsum
            for (p in 0 until N) for (ch in 0 until 3) {
                img.data[p * 3 + ch] = (acc[ch * N + p] * inv + 0.5f).toInt().coerceIn(0, 255).toByte()
            }
            return img
        }
    }

    fun infer(x: Net.Tensor): Net.Tensor =
        (if (x.shape[1].toInt() == T) net ?: throw IllegalStateException("16-frame restorer not loaded") else short).run("frames" to x)

    /** A photo: the crop shown as 4 identical frames; the 4 results are averaged. */
    fun still(crop: Img): Img {
        val out = infer(input(listOf(toCrop(crop)), T_SHORT))
        val acc = FloatArray(3 * N)
        for (i in 0 until T_SHORT) accumulate(out, i, 1f, acc)
        return toImg(acc, T_SHORT.toFloat())
    }
}
