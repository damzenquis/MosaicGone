package com.mosaicgone.app

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.MediaStore
import android.provider.OpenableColumns
import com.mosaicgone.core.ImageOps
import com.mosaicgone.core.Img
import java.io.File
import java.io.IOException
import kotlin.math.max
import kotlin.math.sqrt

object Media {

    /** Only for on-screen previews (before/after on the result screen); jobs use the full picture. */
    const val PREVIEW_PIXELS = 16_000_000
    fun displayName(ctx: Context, uri: Uri): String {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0)?.let { return it }
        }
        return uri.lastPathSegment ?: "media"
    }

    fun isVideo(ctx: Context, uri: Uri) = mime(ctx, uri).startsWith("video/")

    /**
     * The file's type: what the provider says, or (files from other apps / folders often come without one, or as
     * application/octet-stream) what its first bytes say.
     */
    fun mime(ctx: Context, uri: Uri): String {
        val t = ctx.contentResolver.getType(uri)
        if (t != null && (t.startsWith("image/") || t.startsWith("video/"))) return t
        return runCatching { sniff(ctx, uri) }.getOrNull() ?: t ?: "application/octet-stream"
    }

    private fun sniff(ctx: Context, uri: Uri): String? {
        val b = ByteArray(16)
        val n = ctx.contentResolver.openInputStream(uri)?.use { s -> var k = 0; while (k < b.size) { val r = s.read(b, k, b.size - k); if (r < 0) break; k += r }; k } ?: return null
        fun ascii(from: Int, len: Int) = if (n >= from + len) String(b, from, len, Charsets.ISO_8859_1) else ""
        fun u(i: Int) = b[i].toInt() and 0xFF
        return when {
            ascii(0, 4) == "GIF8" -> "image/gif"
            n >= 3 && u(0) == 0xFF && u(1) == 0xD8 && u(2) == 0xFF -> "image/jpeg"
            n >= 8 && u(0) == 0x89 && ascii(1, 3) == "PNG" -> "image/png"
            ascii(0, 4) == "RIFF" && ascii(8, 4) == "WEBP" -> "image/webp"
            ascii(0, 4) == "RIFF" && ascii(8, 4) == "AVI " -> "video/x-msvideo"
            ascii(0, 2) == "BM" -> "image/bmp"
            ascii(4, 4) == "ftyp" -> when (ascii(8, 4)) {
                "heic", "heix", "hevc", "heim", "heis", "mif1", "msf1" -> "image/heic"
                "avif", "avis" -> "image/avif"
                "qt  " -> "video/quicktime"
                "3gp4", "3gp5", "3gp6", "3g2a" -> "video/3gpp"
                else -> "video/mp4"
            }
            n >= 4 && u(0) == 0x1A && u(1) == 0x45 && u(2) == 0xDF && u(3) == 0xA3 -> "video/x-matroska"
            n >= 1 && u(0) == 0x47 -> "video/mp2t"
            else -> null
        }
    }

    fun isGif(ctx: Context, uri: Uri) = mime(ctx, uri) == "image/gif"

    fun readBytes(ctx: Context, uri: Uri): ByteArray =
        ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: throw IOException("Cannot read the file")

    fun fileSize(ctx: Context, uri: Uri): Long {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) return c.getLong(0)
        }
        return ctx.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
    }

    private val KEEP_SPACES = setOf(
        ColorSpace.get(ColorSpace.Named.DISPLAY_P3), ColorSpace.get(ColorSpace.Named.DCI_P3), ColorSpace.get(ColorSpace.Named.ADOBE_RGB),
    )

    /** A decoded photo plus what is needed to save the result like the source. */
    class Photo(val bitmap: Bitmap, val mime: String, val sourceWidth: Int, val sourceHeight: Int, val sourceBytes: Long) {
        // read once while the bitmap is alive: the job recycles it right after converting it, and saving needs this later
        val colorSpace: ColorSpace = bitmap.colorSpace ?: ColorSpace.get(ColorSpace.Named.SRGB)

        /** bits per pixel of the source file */
        val sourceBpp: Double get() = if (sourceBytes > 0) sourceBytes * 8.0 / (sourceWidth.toLong() * sourceHeight) else 0.0
    }

    /** Decodes keeping the source's colour space (e.g. Display P3), so the result can be tagged with the same one. */
    fun decodePhoto(ctx: Context, uri: Uri, maxPixels: Int): Photo {
        var sw = 0; var sh = 0
        val src = ImageDecoder.createSource(ctx.contentResolver, uri)
        var bmp = ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            sw = info.size.width; sh = info.size.height
            if (sw.toLong() * sh > maxPixels) {
                val s = sqrt(maxPixels.toDouble() / (sw.toLong() * sh))
                decoder.setTargetSize(max(1, (sw * s).toInt()), max(1, (sh * s).toInt()))
            }
            // common 8-bit wide-gamut spaces are kept (and tagged in the result); anything else, HDR included, becomes sRGB
            val cs = info.colorSpace
            decoder.setTargetColorSpace(if (cs != null && cs in KEEP_SPACES) cs else ColorSpace.get(ColorSpace.Named.SRGB))
        }
        if (bmp.config != Bitmap.Config.ARGB_8888) {
            // e.g. a 16-bit PNG (8 bytes/pixel): free the original as soon as the 8-bit copy exists, as decodeBitmap does
            val copy = bmp.copy(Bitmap.Config.ARGB_8888, false)
            bmp.recycle()
            bmp = copy
        }
        return try {
            Photo(bmp, mime(ctx, uri), sw, sh, fileSize(ctx, uri))
        } catch (e: Exception) {
            bmp.recycle() // the caller never gets it
            throw e
        }
    }

    fun decodeBitmap(ctx: Context, uri: Uri, maxPixels: Int = PREVIEW_PIXELS, maxSide: Int = 0): Bitmap {
        val src = ImageDecoder.createSource(ctx.contentResolver, uri)
        val bmp = ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val w = info.size.width
            val h = info.size.height
            var s = 1.0
            if (w.toLong() * h > maxPixels) s = sqrt(maxPixels.toDouble() / (w.toLong() * h))
            if (maxSide > 0 && max(w, h) * s > maxSide) s = maxSide.toDouble() / max(w, h)
            if (s < 1.0) decoder.setTargetSize(max(1, (w * s).toInt()), max(1, (h * s).toInt()))
        }
        return if (bmp.config == Bitmap.Config.ARGB_8888) bmp
        else bmp.copy(Bitmap.Config.ARGB_8888, false).also { bmp.recycle() }
    }

    fun toImg(bmp: Bitmap): Img {
        val img = Img(bmp.width, bmp.height, 3)
        val row = IntArray(bmp.width)
        var q = 0
        for (y in 0 until bmp.height) {
            bmp.getPixels(row, 0, bmp.width, 0, y, bmp.width, 1)
            for (px in row) {
                img.data[q] = (px shr 16).toByte(); img.data[q + 1] = (px shr 8).toByte(); img.data[q + 2] = px.toByte()
                q += 3
            }
        }
        return img
    }

    fun toBitmap(img: Img, colorSpace: ColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)): Bitmap {
        val bmp = Bitmap.createBitmap(img.w, img.h, Bitmap.Config.ARGB_8888, false, colorSpace)
        val row = IntArray(img.w)
        for (y in 0 until img.h) {
            for (x in 0 until img.w) {
                row[x] = if (img.c == 1) {
                    val v = img[x, y]
                    (0xFF shl 24) or (v * 0x10101)
                } else (0xFF shl 24) or (img[x, y, 0] shl 16) or (img[x, y, 1] shl 8) or img[x, y, 2]
            }
            bmp.setPixels(row, 0, img.w, 0, y, img.w, 1)
        }
        return bmp
    }

    /** Small copy for the live preview. */
    fun previewBitmap(img: Img, maxSide: Int = 480): Bitmap {
        val s = maxSide.toDouble() / max(img.w, img.h)
        val small = if (s < 1) ImageOps.resize(img, max(1, (img.w * s).toInt()), max(1, (img.h * s).toInt())) else img
        return toBitmap(small)
    }

    private fun insert(ctx: Context, collection: Uri, name: String, mime: String, dir: String): Uri {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            put(MediaStore.MediaColumns.RELATIVE_PATH, dir)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        return ctx.contentResolver.insert(collection, values) ?: throw IOException("Could not create $dir/$name")
    }

    private fun publish(ctx: Context, uri: Uri) {
        ctx.contentResolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
    }

    class Saved(val uri: Uri, val name: String, val note: String)

    /**
     * Saves to Pictures/MosaicGone (the phone's version of ./result) in the source's format and at the source's
     * bits per pixel: PNG stays lossless PNG; JPEG and WebP get the lowest quality whose file is at least as dense
     * as the source. Formats Android cannot write (HEIC/HEIF/AVIF...) become JPEG at 2x the source's bits per pixel,
     * because JPEG needs about twice the bits of HEIC for the same quality.
     */
    fun savePhoto(ctx: Context, img: Img, base: String, src: Photo): Saved {
        val bmp = toBitmap(img, src.colorSpace)
        try {
            val (format, ext, mime, factor) = when (src.mime) {
                "image/png" -> Quad(Bitmap.CompressFormat.PNG, "png", "image/png", 1.0)
                "image/webp" -> Quad(
                    if (android.os.Build.VERSION.SDK_INT >= 30) Bitmap.CompressFormat.WEBP_LOSSY
                    else @Suppress("DEPRECATION") Bitmap.CompressFormat.WEBP, // lossy below quality 100 on Android 10
                    "webp", "image/webp", 1.0,
                )
                "image/jpeg" -> Quad(Bitmap.CompressFormat.JPEG, "jpg", "image/jpeg", 1.0)
                else -> Quad(Bitmap.CompressFormat.JPEG, "jpg", "image/jpeg", 2.0)
            }
            val targetBpp = src.sourceBpp * factor
            val bytes: ByteArray
            val note: String
            if (format == Bitmap.CompressFormat.PNG) {
                bytes = encode(bmp, format, 100)
                note = "PNG (lossless)"
            } else {
                val target = (targetBpp * img.w * img.h / 8).toLong()
                var lo = 40; var hi = 100
                var best = encode(bmp, format, 100); var bestQ = 100
                if (target > 0 && best.size > target) {
                    while (lo <= hi) {
                        val q = (lo + hi) / 2
                        val b = encode(bmp, format, q)
                        if (b.size >= target) { best = b; bestQ = q; hi = q - 1 } else lo = q + 1
                    }
                }
                bytes = best
                note = String.format(
                    java.util.Locale.US, "%s quality %d: %.2f bits/pixel (source %s %.2f bits/pixel%s)",
                    ext.uppercase(), bestQ, bytes.size * 8.0 / (img.w.toLong() * img.h), src.mime.removePrefix("image/").uppercase(), src.sourceBpp,
                    if (factor != 1.0) ", x2 for JPEG" else "",
                )
            }
            val name = "$base.$ext"
            val uri = insert(ctx, MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), name, mime, "Pictures/MosaicGone")
            try {
                ctx.contentResolver.openOutputStream(uri)!!.use { it.write(bytes) }
            } catch (e: Exception) {
                ctx.contentResolver.delete(uri, null, null)
                throw e
            }
            publish(ctx, uri)
            val cs = if (src.colorSpace.isSrgb) "" else " · colour space ${src.colorSpace.name}"
            return Saved(uri, name, note + cs)
        } finally {
            bmp.recycle()
        }
    }

    private data class Quad(val format: Bitmap.CompressFormat, val ext: String, val mime: String, val factor: Double)

    private fun encode(bmp: Bitmap, format: Bitmap.CompressFormat, q: Int): ByteArray =
        java.io.ByteArrayOutputStream(1 shl 20).also { bmp.compress(format, q, it) }.toByteArray()

    fun saveGif(ctx: Context, bytes: ByteArray, name: String): Uri {
        val uri = insert(ctx, MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), name, "image/gif", "Pictures/MosaicGone")
        try {
            ctx.contentResolver.openOutputStream(uri)!!.use { it.write(bytes) }
        } catch (e: Exception) {
            ctx.contentResolver.delete(uri, null, null)
            throw e
        }
        publish(ctx, uri)
        return uri
    }

    const val LOG_DIR = "Download/MosaicGone"

    /** Saves the job log as UTF-8 text to Download/MosaicGone. */
    fun saveText(ctx: Context, text: String, name: String): Uri {
        val uri = insert(ctx, MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), name, "text/plain", LOG_DIR)
        try {
            ctx.contentResolver.openOutputStream(uri)!!.use { it.write(text.toByteArray(Charsets.UTF_8)) }
        } catch (e: Exception) {
            ctx.contentResolver.delete(uri, null, null)
            throw e
        }
        publish(ctx, uri)
        return uri
    }

    /** Copies the finished mp4 to Movies/MosaicGone. */
    fun saveVideo(ctx: Context, file: File, name: String): Uri {
        val uri = insert(ctx, MediaStore.Video.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), name, "video/mp4", "Movies/MosaicGone")
        try {
            ctx.contentResolver.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out, 1 shl 20) } }
        } catch (e: Exception) {
            ctx.contentResolver.delete(uri, null, null)
            throw e
        }
        publish(ctx, uri)
        return uri
    }
}
