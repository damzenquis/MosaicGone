package com.mosaicgone.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.min

/** One ONNX model. Thread-confined: use from one worker thread. */
class Net private constructor(private val session: OrtSession) : AutoCloseable {

    class Tensor(val data: FloatArray, val shape: LongArray)

    fun run(vararg inputs: Pair<String, Tensor>): Tensor {
        val env = OrtEnvironment.getEnvironment()
        val ort = inputs.associate { (name, t) -> name to OnnxTensor.createTensor(env, FloatBuffer.wrap(t.data), t.shape) }
        try {
            session.run(ort).use { result ->
                val out = result.get(0) as OnnxTensor
                val fb = out.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                return Tensor(arr, out.info.shape)
            }
        } finally {
            ort.values.forEach { it.close() }
        }
    }

    override fun close() = session.close()

    companion object {
        private fun options(threads: Int) = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // idle worker threads sleep instead of busy-waiting between operators: less heat, same throughput here
            addConfigEntry("session.intra_op.allow_spinning", "0")
            addConfigEntry("session.inter_op.allow_spinning", "0")
        }

        fun fromFile(path: String, threads: Int = 4) =
            Net(OrtEnvironment.getEnvironment().createSession(path, options(threads)))

        fun fromBytes(bytes: ByteArray, threads: Int = 4) =
            Net(OrtEnvironment.getEnvironment().createSession(bytes, options(threads)))

        /** [buffer] must be direct, e.g. a memory-mapped uncompressed APK asset. */
        fun fromBuffer(buffer: java.nio.ByteBuffer, threads: Int = 4) =
            Net(OrtEnvironment.getEnvironment().createSession(buffer, options(threads)))
    }
}

/** What finds the mosaic: the YOLOX-Nano detector; its output is a 0..255 mask at short side 360. */
class MaskNet(val net: Net) {
    fun input(img: Img): Net.Tensor = Nets.detInput(img)
    fun output(w: Int, h: Int, t: Net.Tensor): Img = Nets.segFromDet(w, h, t)
    fun run(t: Net.Tensor): Net.Tensor = net.run("input" to t)
}

/** Tensor conversions for the detector and restorer, and the mosaic square from the detector's mask. */
object Nets {

    // ------------------------------------------------------------------ MosaicGone's models

    /** mosaicgone_detector input side: the frame letterboxed into DET_SIZE x DET_SIZE (long side fits, pad grey 114) */
    const val DET_SIZE = 416
    /** the detector's mask is at stride 4 */
    const val DET_STRIDE = 4
    /** mosaicgone_restorer crop side */
    const val CROP = 256

    /** Letterboxed, BGR planes, 0..255 (YOLOX convention), area-shrunk like the training data (cv2 INTER_AREA). */
    fun detInput(img: Img): Net.Tensor {
        val r = minOf(DET_SIZE.toDouble() / img.w, DET_SIZE.toDouble() / img.h)
        val nw = maxOf(1, Math.round(img.w * r).toInt())
        val nh = maxOf(1, Math.round(img.h * r).toInt())
        val small = if (r < 1) ImageOps.resizeArea(img, nw, nh) else ImageOps.resize(img, nw, nh)
        val n = DET_SIZE * DET_SIZE
        val out = FloatArray(3 * n) { 114f }
        for (y in 0 until nh) for (x in 0 until nw) {
            val p = (y * nw + x) * 3
            val q = y * DET_SIZE + x
            out[q] = (small.data[p + 2].toInt() and 0xFF).toFloat()
            out[n + q] = (small.data[p + 1].toInt() and 0xFF).toFloat()
            out[2 * n + q] = (small.data[p].toInt() and 0xFF).toFloat()
        }
        return Net.Tensor(out, longArrayOf(1, 3, DET_SIZE.toLong(), DET_SIZE.toLong()))
    }

    /**
     * The detector's (1,1,104,104) probability map -> a 0..255 mask at short side 360 of the frame (letterbox padding
     * cut off), for the threshold, largest area, bounding square and feathered paste.
     */
    fun segFromDet(w: Int, h: Int, t: Net.Tensor): Img {
        val side = t.shape[3].toInt()
        val r = minOf(DET_SIZE.toDouble() / w, DET_SIZE.toDouble() / h)
        // the part of the map that covers the frame (the rest is letterbox padding)
        val vw = (w * r / DET_STRIDE).coerceIn(1.0, side.toDouble())
        val vh = (h * r / DET_STRIDE).coerceIn(1.0, side.toDouble())
        val cw = kotlin.math.ceil(vw).toInt()
        val ch = kotlin.math.ceil(vh).toInt()
        val m = Img(cw, ch, 1)
        for (y in 0 until ch) for (x in 0 until cw) m.data[y * cw + x] = (t.data[y * side + x] * 255f).coerceIn(0f, 255f).toInt().toByte()
        val (sw, sh) = if (w >= h) maxOf(1, (360.0 * w / h).toInt()) to 360 else 360 to maxOf(1, (360.0 * h / w).toInt())
        // scale so the frame's part (vw x vh, fractional) maps onto sw x sh
        val fullW = maxOf(sw, Math.round(sw * cw / vw).toInt())
        val fullH = maxOf(sh, Math.round(sh * ch / vh).toInt())
        return ImageOps.resizeRegion(m, fullW, fullH, 0, 0, sw, sh, ImageOps.Interp.LINEAR)
    }

    /** One restorer crop as RGB planes 0..1 into frame slot [t] of a (1, T, 3, CROP, CROP) input. */
    fun fillRestorer(crop: Img, dst: FloatArray, t: Int) {
        val n = CROP * CROP
        val base = t * 3 * n
        for (p in 0 until n) for (ch in 0 until 3) dst[base + ch * n + p] = (crop.data[p * 3 + ch].toInt() and 0xFF) / 255f
    }

    class MosaicPos(val x: Int, val y: Int, val size: Int, val mask: Img)

    /** The mosaic square of a w x h frame from the detector's mask (the mask stays at short side 360). */
    fun mosaicPosFromSeg(w: Int, h: Int, seg: Img, opt: Options): MosaicPos {
        var mask = ImageOps.maskThreshold(seg, min(h, w) / 20, opt.maskThreshold)
        if (!opt.allMosaicArea) mask = ImageOps.findMostLikelyROI(mask)
        val b = ImageOps.boundingSquare(mask, opt.exMultValue)
        val rat = min(h, w) / 360.0
        var x = (rat * b[0]).toInt().coerceIn(0, w)
        var y = (rat * b[1]).toInt().coerceIn(0, h)
        var size = (rat * b[2]).toInt()
        // keep the square inside the frame on every side
        size = size.coerceIn(0, minOf(w - x, h - y, x, y))
        return MosaicPos(x, y, size, mask)
    }
}
