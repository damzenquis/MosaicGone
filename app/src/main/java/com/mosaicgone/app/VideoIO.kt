package com.mosaicgone.app

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import android.os.Build
import com.mosaicgone.core.FrameSink
import com.mosaicgone.core.FrameSource
import com.mosaicgone.core.ImageOps
import com.mosaicgone.core.Img
import com.mosaicgone.core.RowPool
import com.mosaicgone.core.Timing
import com.mosaicgone.core.Yuv
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Colour description of the source, shared by the decoder (which may refine it from its output format)
 * and the encoder (which tags the result with the same values).
 */
class ColorInfo(@Volatile var spec: Yuv.Spec, @Volatile var known: Boolean, @Volatile var standardTag: Int, @Volatile var rangeTag: Int) {

    fun update(f: MediaFormat, w: Int, h: Int) {
        val std = f.int(MediaFormat.KEY_COLOR_STANDARD)
        val rng = f.int(MediaFormat.KEY_COLOR_RANGE)
        val tf = f.int(MediaFormat.KEY_COLOR_TRANSFER)
        if (std == 0 && rng == 0 && tf == 0) return
        // keep what the container said for anything the decoder leaves out
        val from = from(if (std != 0) std else standardTag, if (rng != 0) rng else rangeTag, if (tf != 0) tf else transferTag(spec.transfer), w, h)
        spec = from.spec; known = true; standardTag = from.standardTag; rangeTag = from.rangeTag
    }

    /** colours of the result: the source's, or SDR BT.709 when the source is HDR (it is tone-mapped on decode) */
    val output: Yuv.Spec get() = spec.sdr
    val outputStandardTag get() = if (spec.isHdr) MediaFormat.COLOR_STANDARD_BT709 else standardTag
    val outputRangeTag get() = if (spec.isHdr) MediaFormat.COLOR_RANGE_LIMITED else rangeTag

    override fun toString() = "$spec" + (if (known) "" else " (not tagged in the file; assumed like players do)") +
        if (spec.isHdr) " → tone-mapped to SDR ${spec.sdr}" else ""

    companion object {
        fun MediaFormat.int(key: String) = if (containsKey(key)) getInteger(key) else 0

        fun transferTag(t: Yuv.Transfer) = when (t) {
            Yuv.Transfer.PQ -> MediaFormat.COLOR_TRANSFER_ST2084
            Yuv.Transfer.HLG -> MediaFormat.COLOR_TRANSFER_HLG
            Yuv.Transfer.SDR -> 0
        }

        fun from(std: Int, rng: Int, tf: Int, w: Int, h: Int): ColorInfo {
            val transfer = when (tf) {
                MediaFormat.COLOR_TRANSFER_ST2084 -> Yuv.Transfer.PQ
                MediaFormat.COLOR_TRANSFER_HLG -> Yuv.Transfer.HLG
                else -> Yuv.Transfer.SDR
            }
            // HDR is BT.2020 in practice, also when the file forgets to say so
            val guess = if (transfer != Yuv.Transfer.SDR) Yuv.Spec(Yuv.Standard.BT2020, false) else Yuv.Spec.guess(w, h)
            val standard = when (std) {
                MediaFormat.COLOR_STANDARD_BT709 -> Yuv.Standard.BT709
                MediaFormat.COLOR_STANDARD_BT601_PAL, MediaFormat.COLOR_STANDARD_BT601_NTSC -> Yuv.Standard.BT601
                MediaFormat.COLOR_STANDARD_BT2020 -> Yuv.Standard.BT2020
                else -> guess.standard
            }
            val full = when (rng) {
                MediaFormat.COLOR_RANGE_FULL -> true
                MediaFormat.COLOR_RANGE_LIMITED -> false
                else -> guess.fullRange
            }
            val stdTag = if (std != 0) std else when (standard) {
                Yuv.Standard.BT709 -> MediaFormat.COLOR_STANDARD_BT709
                Yuv.Standard.BT2020 -> MediaFormat.COLOR_STANDARD_BT2020
                else -> MediaFormat.COLOR_STANDARD_BT601_NTSC
            }
            return ColorInfo(Yuv.Spec(standard, full, transfer), std != 0 || rng != 0 || tf != 0, stdTag,
                if (full) MediaFormat.COLOR_RANGE_FULL else MediaFormat.COLOR_RANGE_LIMITED)
        }
    }
}

class VideoInfo(
    val width: Int, val height: Int, val rotation: Int, val fps: Double, val durationUs: Long, val hasAudio: Boolean,
    val mime: String,
    /** bits per second of the video track (what the result is encoded with) */
    val videoBitrate: Int,
    val bitrateSource: String,
    val color: ColorInfo,
    /** 10-bit (or deeper) source, from the codec profile; decoded at full precision where the phone can */
    val tenBit: Boolean = false,
) {
    val displayWidth get() = if (rotation % 180 == 0) width else height
    val displayHeight get() = if (rotation % 180 == 0) height else width

    /** Processing size: upright, short side capped at [maxShort] (0 = original), even numbers for the encoder. */
    fun outputSize(maxShort: Int): Pair<Int, Int> {
        var w = displayWidth.toDouble()
        var h = displayHeight.toDouble()
        val short = min(w, h)
        if (maxShort > 0 && short > maxShort) {
            val s = maxShort / short
            w *= s; h *= s
        }
        return (w.roundToInt() and 1.inv()).coerceAtLeast(2) to (h.roundToInt() and 1.inv()).coerceAtLeast(2)
    }
}

private fun MediaExtractor.findTrack(prefix: String): Int? {
    for (i in 0 until trackCount) {
        if (getTrackFormat(i).getString(MediaFormat.KEY_MIME)?.startsWith(prefix) == true) return i
    }
    return null
}

object VideoProbe {
    fun probe(ctx: Context, uri: Uri): VideoInfo {
        val ex = MediaExtractor()
        val mmr = MediaMetadataRetriever()
        try {
            try {
                ex.setDataSource(ctx, uri, null)
            } catch (e: Exception) {
                throw IOException(
                    "This phone's video reader cannot open this file (${e.message ?: e.javaClass.simpleName}). " +
                        "Android reads MP4 / MOV / 3GP, MKV / WebM and MPEG-TS; other containers (e.g. AVI, WMV, FLV) need converting first.", e,
                )
            }
            val vt = ex.findTrack("video/") ?: throw IOException("This file has no video track.")
            val f = ex.getTrackFormat(vt)
            mmr.setDataSource(ctx, uri)
            val rotation = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            val durationUs = if (f.containsKey(MediaFormat.KEY_DURATION)) f.getLong(MediaFormat.KEY_DURATION)
            else (mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L) * 1000
            var fps = if (f.containsKey(MediaFormat.KEY_FRAME_RATE)) f.getNumber(MediaFormat.KEY_FRAME_RATE)?.toDouble() ?: 0.0 else 0.0
            if (fps <= 0 || fps > 240) {
                val frames = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toDoubleOrNull() ?: 0.0
                fps = if (frames > 0 && durationUs > 0) frames / (durationUs / 1e6) else 30.0
            }
            val w = f.getInteger(MediaFormat.KEY_WIDTH)
            val h = f.getInteger(MediaFormat.KEY_HEIGHT)
            // HDR (PQ / HLG) is tone-mapped to SDR on decode
            val color = with(ColorInfo) {
                ColorInfo.from(f.int(MediaFormat.KEY_COLOR_STANDARD), f.int(MediaFormat.KEY_COLOR_RANGE), f.int(MediaFormat.KEY_COLOR_TRANSFER), w, h)
            }
            val mime = f.getString(MediaFormat.KEY_MIME) ?: MediaFormat.MIMETYPE_VIDEO_AVC
            // the video track's own bitrate; else the file's minus the audio track's
            val at = ex.findTrack("audio/")
            val audioBps = at?.let { i -> ex.getTrackFormat(i).let { if (it.containsKey(MediaFormat.KEY_BIT_RATE)) it.getInteger(MediaFormat.KEY_BIT_RATE) else 128_000 } } ?: 0
            var bps = if (f.containsKey(MediaFormat.KEY_BIT_RATE)) f.getInteger(MediaFormat.KEY_BIT_RATE) else 0
            var from = "video track"
            if (bps <= 0) {
                val total = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toIntOrNull() ?: 0
                if (total > 0) { bps = total - audioBps; from = "file bitrate minus audio" }
            }
            if (bps <= 0 && durationUs > 0) {
                val len = ctx.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
                if (len > 0) { bps = (len * 8 / (durationUs / 1e6)).toInt() - audioBps; from = "file size / duration minus audio" }
            }
            if (bps <= 0) { bps = (w.toLong() * h * fps * 0.2).toInt(); from = "unknown, estimated" }
            return VideoInfo(
                w, h, ((rotation % 360) + 360) % 360, fps, durationUs, at != null,
                mime, max(100_000, bps), from, color, color.spec.isHdr || isTenBitProfile(mime, with(ColorInfo) { f.int(MediaFormat.KEY_PROFILE) }),
            )
        } finally {
            ex.release()
            mmr.release()
        }
    }

    private fun isTenBitProfile(mime: String, profile: Int): Boolean {
        val p = MediaCodecInfo.CodecProfileLevel::class.java
        fun c(name: String) = runCatching { p.getField(name).getInt(null) }.getOrNull()
        val names = when (mime) {
            MediaFormat.MIMETYPE_VIDEO_HEVC -> listOf("HEVCProfileMain10", "HEVCProfileMain10HDR10", "HEVCProfileMain10HDR10Plus")
            MediaFormat.MIMETYPE_VIDEO_VP9 -> listOf("VP9Profile2", "VP9Profile3", "VP9Profile2HDR", "VP9Profile3HDR", "VP9Profile2HDR10Plus", "VP9Profile3HDR10Plus")
            MediaFormat.MIMETYPE_VIDEO_AV1 -> listOf("AV1ProfileMain10", "AV1ProfileMain10HDR10", "AV1ProfileMain10HDR10Plus")
            MediaFormat.MIMETYPE_VIDEO_AVC -> listOf("AVCProfileHigh10")
            else -> emptyList()
        }
        return profile != 0 && names.any { c(it) == profile }
    }
}

/**
 * Decodes frames in presentation order, upright and at [outW]x[outH] (area-averaged when smaller than the source).
 * Keeps frames in [startUs, endUs) (endUs = 0: to the end), thinned to [fps] when it is lower than the source.
 * Records each returned frame's timestamp (relative to startUs) in [pts], so the encoder can reuse them.
 */
class VideoFrameSource(
    ctx: Context, uri: Uri, private val info: VideoInfo,
    private val outW: Int, private val outH: Int,
    private val startUs: Long, private val endUs: Long, fps: Int,
    private val pts: ArrayList<Long>,
    private val timing: Timing? = null,
    /** log; [announce]: also say which decoder is used (once per job) */
    private val log: ((String) -> Unit)? = null,
    /** --parallel_color helper threads */
    private val rows: RowPool? = null,
    announce: Boolean = true,
) : FrameSource {
    private val ex = MediaExtractor()
    private var codec: MediaCodec
    private val format: MediaFormat
    private val mime: String
    /** presentation time of the last frame handed out (-1 = none yet), to resume after the decoder is lost */
    private var lastUs = -1L
    private var recoveries = 0
    private val intervalUs = if (fps > 0 && fps < info.fps - 0.01) 1e6 / fps else 0.0
    private var nextKeepUs = startUs.toDouble()
    private var inputDone = false
    private var outputDone = false
    private var index = 0
    private val bi = MediaCodec.BufferInfo()
    private val planes = Array(3) { ByteArray(0) }

    override val estimatedFrames: Int

    init {
        // if opening fails, nobody gets an object to close(): release the extractor (and a decoder already started) here
        var opened: MediaCodec? = null
        try {
            ex.setDataSource(ctx, uri, null)
            val track = ex.findTrack("video/") ?: throw IOException("This file has no video track.")
            ex.selectTrack(track)
            format = ex.getTrackFormat(track)
            mime = format.getString(MediaFormat.KEY_MIME)!!
            codec = openDecoder(softwareFirst = false, announce = announce).also { opened = it }
            if (startUs > 0) ex.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            val end = if (endUs > 0) min(endUs, info.durationUs) else info.durationUs
            val outFps = if (intervalUs > 0) 1e6 / intervalUs else info.fps
            estimatedFrames = max(1, ((end - startUs) / 1e6 * outFps).roundToInt())
        } catch (t: Throwable) {
            opened?.let { c -> runCatching { c.stop() }; runCatching { c.release() } }
            runCatching { ex.release() }
            throw t
        }
    }

    /**
     * Every decoder that claims this format, the platform's choice first (or software ones first, after Android took
     * a hardware decoder away); one that refuses is skipped.
     */
    private fun openDecoder(softwareFirst: Boolean, announce: Boolean): MediaCodec {
        val all = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { !it.isEncoder && mime in it.supportedTypes }
        val preferred = runCatching {
            MediaCodecList(MediaCodecList.REGULAR_CODECS).findDecoderForFormat(MediaFormat(format).apply { removeKey(MediaFormat.KEY_FRAME_RATE) })
        }.getOrNull()
        var candidates = (listOfNotNull(all.firstOrNull { it.name == preferred }) + all.filter { it.name != preferred && !it.isSoftwareOnly } +
            all.filter { it.name != preferred && it.isSoftwareOnly }).distinctBy { it.name }
        if (softwareFirst) candidates = candidates.filter { it.isSoftwareOnly } + candidates.filter { !it.isSoftwareOnly }
        if (candidates.isEmpty()) throw IOException("This phone has no decoder for this video's format ($mime).")
        var opened: MediaCodec? = null
        var lastError: Exception? = null
        for (ci in candidates) {
            // 10-bit sources: ask for 10-bit output (P010) where offered, so HDR tone mapping keeps the precision
            // lint (InlinedApi): a compile-time constant, copied into the app; it is only used when SDK_INT >= 33 (below)
            @SuppressLint("InlinedApi")
            val p010 = MediaCodecInfo.CodecCapabilities.COLOR_FormatYUVP010
            val wantP010 = info.tenBit && Build.VERSION.SDK_INT >= 33 &&
                runCatching { p010 in ci.getCapabilitiesForType(mime).colorFormats }.getOrDefault(false)
            for (colorFormat in listOfNotNull(if (wantP010) p010 else null, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)) {
                val c = runCatching { MediaCodec.createByCodecName(ci.name) }.getOrElse { lastError = it as? Exception; null } ?: break
                try {
                    c.configure(MediaFormat(format).apply { setInteger(MediaFormat.KEY_COLOR_FORMAT, colorFormat) }, null, null, 0)
                    c.start()
                    opened = c
                    if (announce) log?.invoke(
                        "   decoder ${ci.name}${if (ci.isHardwareAccelerated) " (hardware)" else " (software)"}" +
                            (if (colorFormat == p010) ", 10-bit output" else "") + if (info.tenBit && colorFormat != p010) ", 10-bit source" else "",
                    )
                    break
                } catch (e: Exception) {
                    lastError = e
                    c.release()
                }
            }
            if (opened != null) break
        }
        return opened ?: throw IOException("No decoder on this phone accepted this video ($mime): ${lastError?.message}")
    }

    /**
     * Android can take a hardware decoder away from a background app when the app on screen needs one (a video
     * player, the camera, a browser playing video). Instead of failing the job: open another decoder (software first,
     * which the system does not reclaim), go back to the keyframe before the next frame we need, and skip what we
     * already handed out.
     */
    private fun recover(e: Exception) {
        recoveries++
        if (recoveries > MAX_RECOVERIES) throw IOException("The video decoder was lost $MAX_RECOVERIES times; giving up (${e.message})", e)
        log?.invoke("   decoder lost (${e.javaClass.simpleName}: ${e.message?.take(80)}); probably another app needed it. Reopening and continuing (#$recoveries)")
        runCatching { codec.release() }
        codec = openDecoder(softwareFirst = true, announce = false)
        log?.invoke("   continuing with decoder ${codec.name}")
        ex.seekTo(max(startUs, lastUs + 1), MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
        inputDone = false
        outputDone = false
    }

    private companion object {
        const val MAX_RECOVERIES = 10
    }

    /** Splits the time of next() into codec work (read.decode) and colour conversion (read.yuv2rgb). */
    override fun next(): Img? {
        val t0 = System.nanoTime()
        var convNanos = 0L
        try {
            return decodeNext { idx ->
                val c0 = System.nanoTime()
                // an Image where the decoder offers one, else its raw buffer (frames used to be skipped silently here)
                val image = codec.getOutputImage(idx)
                val f = if (image != null) image.use { convert(it) } else convertBuffer(codec.getOutputBuffer(idx)!!.apply { position(bi.offset); limit(bi.offset + bi.size) })
                convNanos = System.nanoTime() - c0
                f
            }
        } finally {
            timing?.add("read.decode", System.nanoTime() - t0 - convNanos)
            if (convNanos > 0) timing?.add("read.yuv2rgb", convNanos)
        }
    }

    private fun copy(buf: ByteBuffer, k: Int): ByteArray {
        val n = buf.remaining()
        if (planes[k].size < n) planes[k] = ByteArray(n)
        buf.get(planes[k], 0, n)
        return planes[k]
    }

    /** source colours -> upright RGB at the processing size */
    private fun convert(image: Image): Img {
        val crop = image.cropRect
        val p = image.planes
        fun plane(k: Int, wide: Boolean) = Yuv.Plane(copy(p[k].buffer, k), p[k].rowStride, p[k].pixelStride, wide = wide)
        fun yuv(subX: Int, subY: Int, wide: Boolean) = Yuv.toRgb(
            plane(0, wide), plane(1, wide), plane(2, wide), crop.left, crop.top, crop.width(), crop.height(), info.color.spec, subX, subY, rows,
        )
        val rgb = when (image.format) {
            ImageFormat.YUV_420_888 -> yuv(2, 2, false)
            ImageFormat.YUV_422_888 -> yuv(2, 1, false)
            ImageFormat.YUV_444_888 -> yuv(1, 1, false)
            ImageFormat.YCBCR_P010 -> yuv(2, 2, true)
            ImageFormat.FLEX_RGB_888, ImageFormat.FLEX_RGBA_8888 -> rgbPlanes(image)
            else -> throw IOException("This phone's decoder gives frames in a layout the app does not read yet (image format ${image.format}).")
        }
        return fit(rgb)
    }

    /** decoders that output RGB directly (rare, e.g. some MJPEG ones) */
    private fun rgbPlanes(image: Image): Img {
        val crop = image.cropRect
        val p = image.planes
        val r = Yuv.Plane(copy(p[0].buffer, 0), p[0].rowStride, p[0].pixelStride)
        val g = Yuv.Plane(copy(p[1].buffer, 1), p[1].rowStride, p[1].pixelStride)
        val b = Yuv.Plane(copy(p[2].buffer, 2), p[2].rowStride, p[2].pixelStride)
        val out = Img(crop.width(), crop.height(), 3)
        var q = 0
        for (y in crop.top until crop.bottom) for (x in crop.left until crop.right) {
            out.data[q++] = r.at(x, y).toInt().toByte(); out.data[q++] = g.at(x, y).toInt().toByte(); out.data[q++] = b.at(x, y).toInt().toByte()
        }
        return out
    }

    /**
     * Decoders that cannot hand out an Image: read the raw buffer with the layout the output format describes
     * (planar I420, semi-planar NV12, or P010).
     */
    private fun convertBuffer(buf: ByteBuffer): Img {
        val f = codec.outputFormat
        fun int(key: String, def: Int) = if (f.containsKey(key)) f.getInteger(key) else def
        val colorFormat = int(MediaFormat.KEY_COLOR_FORMAT, 0)
        val w = int(MediaFormat.KEY_WIDTH, info.width)
        val h = int(MediaFormat.KEY_HEIGHT, info.height)
        val stride = int(MediaFormat.KEY_STRIDE, w).let { if (it <= 0) w else it }
        val slice = int(MediaFormat.KEY_SLICE_HEIGHT, h).let { if (it <= 0) h else it }
        val left = int("crop-left", 0); val top = int("crop-top", 0)
        val cw = int("crop-right", w - 1) - left + 1
        val chh = int("crop-bottom", h - 1) - top + 1
        val data = copy(buf, 0)
        val cc = MediaCodecInfo.CodecCapabilities::class.java
        val rgb = when (colorFormat) {
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420PackedPlanar -> {
                val ySize = stride * slice
                val cStride = stride / 2
                Yuv.toRgb(
                    Yuv.Plane(data, stride, 1), Yuv.Plane(data, cStride, 1, ySize), Yuv.Plane(data, cStride, 1, ySize + cStride * (slice / 2)),
                    left, top, cw, chh, info.color.spec, rows = rows,
                )
            }
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420PackedSemiPlanar -> {
                val ySize = stride * slice
                Yuv.toRgb(Yuv.Plane(data, stride, 1), Yuv.Plane(data, stride, 2, ySize), Yuv.Plane(data, stride, 2, ySize + 1), left, top, cw, chh, info.color.spec, rows = rows)
            }
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUVP010 -> {
                // decoders report this stride in bytes or in pixels; a row of 16-bit samples needs at least 2 bytes per pixel
                val rowBytes = if (stride < w * 2) stride * 2 else stride
                val ySize = rowBytes * slice
                Yuv.toRgb(
                    Yuv.Plane(data, rowBytes, 2, 0, true), Yuv.Plane(data, rowBytes, 4, ySize, true), Yuv.Plane(data, rowBytes, 4, ySize + 2, true),
                    left, top, cw, chh, info.color.spec, rows = rows,
                )
            }
            else -> throw IOException(
                "This phone's decoder gives frames in a layout the app does not read yet (colour format $colorFormat" +
                    (cc.fields.firstOrNull { runCatching { it.getInt(null) == colorFormat && it.name.startsWith("COLOR_") }.getOrDefault(false) }?.let { ", ${it.name}" } ?: "") + ").",
            )
        }
        return fit(rgb)
    }

    private fun fit(rgb: Img): Img {
        val up = Yuv.rotate(rgb, info.rotation)
        return when {
            up.w == outW && up.h == outH -> up
            // odd source sizes: drop the last row/column instead of resampling the whole frame
            up.w - outW in 0..1 && up.h - outH in 0..1 -> up.crop(0, 0, outW, outH)
            else -> ImageOps.resizeArea(up, outW, outH)
        }
    }

    private inline fun decodeNext(convert: (Int) -> Img): Img? {
        while (!outputDone) {
            try {
            if (!inputDone) {
                val inIdx = codec.dequeueInputBuffer(10_000)
                if (inIdx >= 0) {
                    val buf = codec.getInputBuffer(inIdx)!!
                    val n = ex.readSampleData(buf, 0)
                    // feed a little past the end so reordered (B) frames still come out
                    if (n < 0 || (endUs > 0 && ex.sampleTime > endUs + 1_000_000)) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inIdx, 0, n, ex.sampleTime, 0)
                        ex.advance()
                    }
                }
            }
            val outIdx = codec.dequeueOutputBuffer(bi, 10_000)
            if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                // the decoder reports the colour description it found in the bitstream
                info.color.update(codec.outputFormat, info.width, info.height)
                continue
            }
            if (outIdx < 0) continue
            if (bi.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
            val t = bi.presentationTimeUs
            var frame: Img? = null
            if (endUs > 0 && t >= endUs) {
                outputDone = true
            } else if (bi.size > 0 && t >= startUs && t > lastUs && (intervalUs == 0.0 || t + 1000 >= nextKeepUs)) {
                if (intervalUs > 0) while (nextKeepUs <= t + 1000) nextKeepUs += intervalUs
                frame = convert(outIdx)
            }
            codec.releaseOutputBuffer(outIdx, false)
            frame?.let {
                lastUs = t
                val rel = max(0L, t - startUs)
                if (index < pts.size) pts[index] = rel else pts.add(rel)
                index++
                return it
            }
            } catch (e: IllegalStateException) {
                // MediaCodec.CodecException is one of these: the decoder was released or broke under us
                recover(e)
            }
        }
        return null
    }

    override fun close() {
        try { codec.stop() } catch (_: Exception) {}
        codec.release()
        ex.release()
    }
}

/**
 * Video encoder + MP4 muxer. Opens on the first frame (its size is the video size).
 * Always encodes at the source's bitrate, in HEVC when the source is HEVC (and the phone can), else H.264 High,
 * with the source's colour standard/range, tagged in the output. Copies the source's audio.
 *
 * If Android takes the encoder away (another app needed it), the part written so far is closed as a finished
 * file, a new encoder continues in a new part (frames the lost encoder still held are sent again), and the parts
 * are joined on [close]. If the job fails, [close] still leaves a playable file of everything written: [outputs].
 */
class VideoFrameSink(
    private val ctx: Context, private val file: File, private val fps: Double,
    private val pts: List<Long>, private val audioUri: Uri?, private val startUs: Long, private val endUs: Long,
    private val log: (String) -> Unit,
    private val timing: Timing? = null,
    private val source: VideoInfo,
    /** --parallel_color helper threads */
    private val rows: RowPool? = null,
) : FrameSink {
    private var codec: MediaCodec? = null
    private var codecMime = ""
    private var codecFormat: MediaFormat? = null
    private var muxer: MediaMuxer? = null
    private var partFile = file
    private val parts = ArrayList<File>()
    private var videoTrack = -1
    private var audioTrack = -1
    private var started = false
    private var w = 0
    private var h = 0
    private var frames = 0
    private var lastPts = -1L
    private val bi = MediaCodec.BufferInfo()
    private var nv12: ByteArray? = null
    private lateinit var spec: Yuv.Spec
    /** frames handed to the encoder but not in the file yet (re-sent if the encoder is lost), by timestamp */
    private val inFlight = LinkedHashMap<Long, Img>()
    /** timestamps written to the current part */
    private var written = 0
    private var totalWritten = 0
    /** result time up to which audio is already in a finished part */
    private var audioDoneUs = 0L
    private var lastWrittenUs = -1L
    private var recoveries = 0
    private var closed = false

    /** files with the result: one normally; several only if parts could not be joined */
    val outputs = ArrayList<File>()

    private fun encoderFor(mime: String, f: MediaFormat): String? = MediaCodecList(MediaCodecList.REGULAR_CODECS).findEncoderForFormat(f)
        ?: MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.firstOrNull { it.isEncoder && mime in it.supportedTypes }?.name

    private fun open(width: Int, height: Int) {
        spec = source.color.output
        var tw = width and 1.inv()
        var th = height and 1.inv()
        val wantHevc = source.mime == MediaFormat.MIMETYPE_VIDEO_HEVC
        val mimes = if (wantHevc) listOf(MediaFormat.MIMETYPE_VIDEO_HEVC, MediaFormat.MIMETYPE_VIDEO_AVC) else listOf(MediaFormat.MIMETYPE_VIDEO_AVC)
        var lastError: Exception? = null
        outer@ for (mime in mimes) for (attempt in 0..2) {
            val f = MediaFormat.createVideoFormat(mime, tw, th).apply {
                setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                setInteger(MediaFormat.KEY_FRAME_RATE, max(1, fps.roundToInt()))
                setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
                setInteger(MediaFormat.KEY_COLOR_STANDARD, source.color.outputStandardTag)
                setInteger(MediaFormat.KEY_COLOR_RANGE, source.color.outputRangeTag)
                setInteger(MediaFormat.KEY_COLOR_TRANSFER, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            }
            val name = encoderFor(mime, f) ?: continue@outer
            val c = MediaCodec.createByCodecName(name)
            val caps = c.codecInfo.getCapabilitiesForType(mime)
            // the source's bitrate (mandatory), only clamped to what this encoder can do
            val range = caps.videoCapabilities?.bitrateRange
            val bps = range?.clamp(source.videoBitrate) ?: source.videoBitrate
            f.setInteger(MediaFormat.KEY_BIT_RATE, bps)
            if (caps.encoderCapabilities?.isBitrateModeSupported(MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR) == true) {
                f.setInteger(MediaFormat.KEY_BITRATE_MODE, MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_VBR)
            }
            val high = mime == MediaFormat.MIMETYPE_VIDEO_AVC && caps.profileLevels.any { it.profile == MediaCodecInfo.CodecProfileLevel.AVCProfileHigh }
            if (high && attempt == 0) f.setInteger(MediaFormat.KEY_PROFILE, MediaCodecInfo.CodecProfileLevel.AVCProfileHigh)
            try {
                c.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                c.start()
                codec = c
                codecMime = mime
                codecFormat = f
                val codecName = if (mime == MediaFormat.MIMETYPE_VIDEO_HEVC) "HEVC" else if (high && attempt == 0) "H.264 High" else "H.264"
                log(
                    String.format(
                        Locale.US, "   encoding %s %dx%d at %.2f Mbps (source %.2f Mbps, %s%s) · colour %s",
                        codecName, tw, th, bps / 1e6, source.videoBitrate / 1e6, source.bitrateSource,
                        if (bps != source.videoBitrate) ", clamped by the encoder" else "", source.color,
                    ),
                )
                if (wantHevc && mime != MediaFormat.MIMETYPE_VIDEO_HEVC) log("   note: source is HEVC but this phone has no HEVC encoder; H.264 at the same bitrate is somewhat less efficient")
                break@outer
            } catch (e: Exception) {
                c.release()
                lastError = e
                // retry without the profile, then with sizes some encoders require (multiples of 16)
                if (attempt == 1) { tw = tw / 16 * 16; th = th / 16 * 16 }
            }
        }
        if (codec == null) throw IOException("No video encoder accepted ${tw}x$th: ${lastError?.message}")
        w = tw; h = th
        muxer = MediaMuxer(partFile.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    }

    /** Splits the time of write() into colour conversion (write.rgb2yuv) and codec/muxer work (write.encode). */
    override fun write(img: Img) {
        val t0 = System.nanoTime()
        convNanos = 0L
        try {
            encode(img)
        } finally {
            timing?.add("write.encode", System.nanoTime() - t0 - convNanos)
            if (convNanos > 0) timing?.add("write.rgb2yuv", convNanos)
        }
    }

    private var convNanos = 0L

    private inline fun <T> conv(block: () -> T): T {
        val c0 = System.nanoTime()
        try {
            return block()
        } finally {
            convNanos += System.nanoTime() - c0
        }
    }

    private fun encode(img: Img) {
        if (codec == null) open(img.w, img.h)
        val f = if (img.w != w || img.h != h) {
            if (img.w >= w && img.h >= h) img.crop(0, 0, w, h) else ImageOps.resize(img, w, h)
        } else img
        var t = if (frames < pts.size) pts[frames] else (lastPts + 1e6 / fps).toLong()
        if (t <= lastPts) t = lastPts + 1
        lastPts = t
        inFlight[t] = f
        frames++
        try {
            feed(f, t)
        } catch (e: IllegalStateException) {
            // MediaCodec.CodecException is one of these: the encoder was released or broke under us
            recoverEncoder(e)
        }
    }

    private fun feed(f: Img, t: Long) {
        val c = codec!!
        var inIdx: Int
        while (true) {
            inIdx = c.dequeueInputBuffer(10_000)
            if (inIdx >= 0) break
            drain(false)
        }
        // read the capacity first: once the Image is taken, the ByteBuffer for this index must not be touched
        val capacity = c.getInputBuffer(inIdx)!!.capacity()
        val image = c.getInputImage(inIdx)
        val yuv = conv { Yuv.fromRgb(f, spec, rows) }
        if (image != null) {
            conv { fillImage(image, yuv) }
            // encoders may pad rows, so the frame can occupy more than w*h*3/2 bytes of the buffer
            c.queueInputBuffer(inIdx, 0, capacity, t, 0)
        } else {
            val buf = c.getInputBuffer(inIdx)!!
            val data = conv { toNv12(yuv) }
            buf.clear()
            buf.put(data)
            c.queueInputBuffer(inIdx, 0, data.size, t, 0)
        }
        drain(false)
    }

    /**
     * The encoder is gone: close the part written so far as a finished file, start a new encoder and part, and send
     * the frames the old encoder still held again.
     */
    private fun recoverEncoder(e: Exception) {
        while (true) {
            recoveries++
            if (recoveries > MAX_RECOVERIES) throw IOException("The video encoder was lost $MAX_RECOVERIES times; giving up (${e.message})", e)
            log("   encoder lost (${e.javaClass.simpleName}: ${e.message?.take(80)}); probably another app needed it. Continuing in a new part (#$recoveries)")
            runCatching { codec?.release() }
            codec = null
            finishPart()
            try {
                reopenEncoder()
                for ((t, img) in inFlight.entries.sortedBy { it.key }.map { it.key to it.value }) feed(img, t)
                return
            } catch (again: IllegalStateException) {
                continue
            }
        }
    }

    /** Same settings; the same encoder first (keeps the stream compatible for joining), then software ones. */
    private fun reopenEncoder() {
        val f = MediaFormat(codecFormat!!)
        val list = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { it.isEncoder && codecMime in it.supportedTypes }
        val previous = encoderFor(codecMime, f)
        val order = (list.filter { it.name == previous } + list.filter { it.isSoftwareOnly } + list).distinctBy { it.name }
        var last: Exception? = null
        for (ci in order) {
            val c = runCatching { MediaCodec.createByCodecName(ci.name) }.getOrNull() ?: continue
            try {
                c.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                c.start()
                codec = c
                log("   continuing with encoder ${ci.name}")
                partFile = File(file.path + ".part${parts.size}")
                partFile.delete()
                muxer = MediaMuxer(partFile.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
                return
            } catch (x: Exception) {
                last = x
                c.release()
            }
        }
        throw IOException("No video encoder could be reopened: ${last?.message}")
    }

    private fun fillImage(image: Image, yuv: Array<ByteArray>) {
        val p = image.planes
        val yb = p[0].buffer
        val yRow = p[0].rowStride
        if (p[0].pixelStride == 1) {
            for (y in 0 until h) { yb.position(y * yRow); yb.put(yuv[0], y * w, w) }
        } else {
            for (y in 0 until h) for (x in 0 until w) yb.put(y * yRow + x * p[0].pixelStride, yuv[0][y * w + x])
        }
        val cw = w / 2
        for ((plane, src) in listOf(p[1] to yuv[1], p[2] to yuv[2])) {
            val b = plane.buffer
            val row = plane.rowStride
            val pix = plane.pixelStride
            if (pix == 1) {
                for (y in 0 until h / 2) { b.position(y * row); b.put(src, y * cw, cw) }
            } else {
                for (y in 0 until h / 2) for (x in 0 until cw) b.put(y * row + x * pix, src[y * cw + x])
            }
        }
    }

    private fun toNv12(yuv: Array<ByteArray>): ByteArray {
        val a = nv12?.takeIf { it.size == w * h * 3 / 2 } ?: ByteArray(w * h * 3 / 2).also { nv12 = it }
        System.arraycopy(yuv[0], 0, a, 0, w * h)
        val cw = w / 2
        for (i in 0 until cw * (h / 2)) {
            a[w * h + 2 * i] = yuv[1][i]
            a[w * h + 2 * i + 1] = yuv[2][i]
        }
        return a
    }

    private fun drain(endOfStream: Boolean) {
        val c = codec ?: return
        while (true) {
            val idx = c.dequeueOutputBuffer(bi, if (endOfStream) 10_000 else 0)
            when {
                idx == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> startMuxer(c.outputFormat)
                idx >= 0 -> {
                    val buf = c.getOutputBuffer(idx)!!
                    if (bi.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) bi.size = 0
                    if (bi.size > 0 && started) {
                        buf.position(bi.offset); buf.limit(bi.offset + bi.size)
                        muxer!!.writeSampleData(videoTrack, buf, bi)
                        written++
                        totalWritten++
                        inFlight.remove(bi.presentationTimeUs)
                        if (bi.presentationTimeUs > lastWrittenUs) lastWrittenUs = bi.presentationTimeUs
                    }
                    c.releaseOutputBuffer(idx, false)
                    if (bi.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    private fun startMuxer(videoFormat: MediaFormat) {
        val m = muxer!!
        videoTrack = m.addTrack(videoFormat)
        audioTrack = -1
        if (audioUri != null) {
            try {
                MediaExtractor().apply { setDataSource(ctx, audioUri, null) }.useExtractor { ex ->
                    ex.findTrack("audio/")?.let { at -> audioTrack = m.addTrack(ex.getTrackFormat(at)) }
                }
            } catch (e: Exception) {
                if (parts.isEmpty()) log("Audio could not be copied (${e.message}); the result will be silent.")
                audioTrack = -1
            }
        }
        m.start()
        started = true
        written = 0
    }

    /** Source audio for result times [fromUs, toUs) (relative to the start of the processed part). */
    private fun copyAudio(m: MediaMuxer, fromUs: Long, toUs: Long) {
        if (audioTrack < 0 || audioUri == null) return
        val a0 = System.nanoTime()
        try {
            MediaExtractor().apply { setDataSource(ctx, audioUri, null) }.useExtractor { ex ->
                val at = ex.findTrack("audio/") ?: return@useExtractor
                ex.selectTrack(at)
                ex.seekTo(startUs + fromUs, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                val buf = ByteBuffer.allocateDirect(1 shl 20)
                val info = MediaCodec.BufferInfo()
                while (true) {
                    val n = ex.readSampleData(buf, 0)
                    if (n < 0) break
                    val t = ex.sampleTime
                    if ((endUs > 0 && t >= endUs) || t - startUs >= toUs) break
                    if (t - startUs >= fromUs) {
                        val flags = if (ex.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                        info.set(0, n, t - startUs, flags)
                        m.writeSampleData(audioTrack, buf, info)
                    }
                    ex.advance()
                }
            }
        } catch (e: Exception) {
            log("Audio copy stopped early: ${e.message}")
        }
        timing?.add("audio", System.nanoTime() - a0)
    }

    /** Ends the current part as a playable file (with its audio) and remembers it. */
    private fun finishPart() {
        val m = muxer ?: return
        muxer = null
        val ok = started && written > 0
        if (ok) {
            // audio up to the end of the last frame of this part; the next part continues from there
            val to = lastWrittenUs + (1e6 / fps).toLong()
            copyAudio(m, audioDoneUs, to)
            audioDoneUs = to
        }
        try { if (started) m.stop() } catch (_: Exception) {}
        m.release()
        started = false
        if (ok) {
            // the first part was written to the final name; keep it aside so the joined file can take that name
            if (partFile == file) {
                val p0 = File(file.path + ".part0")
                p0.delete()
                if (!file.renameTo(p0)) throw IOException("Could not keep the first part of the video")
                partFile = p0
            }
            parts.add(partFile)
        } else partFile.delete()
    }

    val frameCount get() = frames

    /** frames actually in the file(s) */
    val writtenCount get() = totalWritten

    override fun close() {
        if (closed) return
        closed = true
        val c = codec
        if (c != null) {
            try {
                var inIdx: Int
                do { inIdx = c.dequeueInputBuffer(10_000); if (inIdx < 0) drain(false) } while (inIdx < 0)
                c.queueInputBuffer(inIdx, 0, 0, lastPts + 1, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                drain(true)
            } catch (e: IllegalStateException) {
                log("   encoder lost while finishing (${e.message?.take(80)}); keeping what was written")
            } finally {
                try { c.stop() } catch (_: Exception) {}
                c.release()
                codec = null
            }
        }
        if (parts.isEmpty()) {
            // the normal case: one part, written to the final name; add the audio and close it
            muxer?.let { m ->
                if (started && written > 0) copyAudio(m, 0, lastWrittenUs + (1e6 / fps).toLong())
                try { if (started) m.stop() } catch (_: Exception) {}
                m.release()
            }
            muxer = null
            if (written > 0) outputs.add(file)
            return
        }
        finishPart()
        join()
    }

    /** Joins the parts into [file]; if their streams differ (a different encoder took over), keeps them as separate files. */
    private fun join() {
        val formats = parts.map { p -> MediaExtractor().apply { setDataSource(p.path) }.useExtractor { ex -> ex.findTrack("video/")?.let { ex.getTrackFormat(it) } } }
        fun csd(f: MediaFormat?, k: String) = f?.takeIf { it.containsKey(k) }?.getByteBuffer(k)?.let { b -> ByteArray(b.remaining()).also { b.duplicate().get(it) } }
        val same = formats.all { f -> csd(f, "csd-0").contentEquals(csd(formats[0], "csd-0")) && csd(f, "csd-1").contentEquals(csd(formats[0], "csd-1")) }
        if (!same) {
            log("   the parts use different encoder settings and cannot be joined; saving them as separate files")
            parts.forEachIndexed { i, p ->
                val dst = if (i == 0) file else File(file.path.removeSuffix(".mp4") + "_part${i + 1}.mp4")
                dst.delete(); p.renameTo(dst); outputs.add(dst)
            }
            return
        }
        val out = MediaMuxer(file.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val tracks = HashMap<String, Int>()
        MediaExtractor().apply { setDataSource(parts[0].path) }.useExtractor { ex ->
            for (i in 0 until ex.trackCount) {
                val f = ex.getTrackFormat(i)
                val kind = f.getString(MediaFormat.KEY_MIME)!!.substringBefore('/')
                tracks[kind] = out.addTrack(f)
            }
        }
        out.start()
        val buf = ByteBuffer.allocateDirect(4 shl 20)
        val info = MediaCodec.BufferInfo()
        for (p in parts) {
            MediaExtractor().apply { setDataSource(p.path) }.useExtractor { ex ->
                val map = HashMap<Int, Int>()
                for (i in 0 until ex.trackCount) {
                    val kind = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME)!!.substringBefore('/')
                    tracks[kind]?.let { map[i] = it; ex.selectTrack(i) }
                }
                while (true) {
                    val n = ex.readSampleData(buf, 0)
                    if (n < 0) break
                    val flags = if (ex.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                    info.set(0, n, ex.sampleTime, flags)
                    map[ex.sampleTrackIndex]?.let { out.writeSampleData(it, buf, info) }
                    ex.advance()
                }
            }
        }
        out.stop()
        out.release()
        parts.forEach { it.delete() }
        outputs.add(file)
        log("   joined ${parts.size} parts into one video")
    }

    private companion object {
        const val MAX_RECOVERIES = 10
    }
}

private inline fun <T> MediaExtractor.useExtractor(block: (MediaExtractor) -> T): T = try { block(this) } finally { release() }

