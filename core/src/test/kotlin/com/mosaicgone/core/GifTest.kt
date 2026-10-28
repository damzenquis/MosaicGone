package com.mosaicgone.core

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class GifTest {
    private val src = File(PipelineTest.images, "shapes.gif")

    @Test fun decodesShapesGif() {
        val a = Gif.decode(src.readBytes())
        assertEquals(768, a.width); assertEquals(214, a.height)
        assertEquals(21, a.frames.size)
        assertEquals(50, a.frames[0].delayCs)
        assertEquals(0, a.loopCount)
        // first frame matches what javax.imageio shows for the first image
        val ref = PipelineTest.load(src)
        var d = 0L
        val first = a.frames[0].image
        for (i in ref.data.indices) d += kotlin.math.abs((ref.data[i].toInt() and 0xFF) - (first.data[i].toInt() and 0xFF))
        assertEquals(0L, d)
    }

    @Test fun roundTripIsLossless() {
        val bytes = src.readBytes()
        val a = Gif.decode(bytes)
        val out = Gif.encode(a, a.frames.map { it.image })
        File(PipelineTest.out, "shapes_roundtrip.gif").writeBytes(out)
        val b = Gif.decode(out)
        assertEquals(a.frames.size, b.frames.size)
        assertEquals(a.loopCount, b.loopCount)
        for (k in a.frames.indices) {
            assertArrayEquals("frame $k", a.frames[k].image.data, b.frames[k].image.data)
            assertEquals(a.frames[k].delayCs, b.frames[k].delayCs)
        }
        println("shapes.gif ${bytes.size} bytes -> re-encoded ${out.size} bytes (%.0f%%)".format(100.0 * out.size / bytes.size))
        assertTrue("re-encoded GIF much larger than the source", out.size < bytes.size * 2)
    }

    @Test fun cleanedGifKeepsSourcePalette() {
        val a = Gif.decode(src.readBytes())
        // the right panel of shapes.gif carries a mosaic; clean it like the app does
        val opt = Options(modelName = Options.RESTORER).resolve()
        val sink = ArrayList<Img>()
        Pipeline(opt, PipelineTest.quiet).cleanMosaicVideoRestorer(
            {
                object : FrameSource {
                    var i = 0
                    override val estimatedFrames = a.frames.size
                    override fun next() = if (i < a.frames.size) a.frames[i++].image else null
                    override fun close() {}
                }
            },
            object : FrameSink {
                override fun write(img: Img) { sink.add(img) }
                override fun close() {}
            },
            PipelineTest.restorer(), PipelineTest.maskNet(),
        )
        val out = Gif.encode(a, sink)
        File(PipelineTest.out, "shapes_clean.gif").writeBytes(out)
        val b = Gif.decode(out)
        assertEquals(21, b.frames.size)
        // every output colour comes from that frame's source palette
        for ((k, f) in b.frames.withIndex()) {
            val pal = a.frames[k].palette.toHashSet()
            val img = f.image
            val opaque = f.opaque
            for (p in 0 until a.width * a.height) {
                val rgb = ((img.data[p * 3].toInt() and 0xFF) shl 16) or ((img.data[p * 3 + 1].toInt() and 0xFF) shl 8) or (img.data[p * 3 + 2].toInt() and 0xFF)
                if (opaque[p] && rgb !in pal) {
                    // a pixel kept from an earlier frame can carry that frame's palette colour
                    assertTrue(b.frames.take(k).any { g -> a.frames[b.frames.indexOf(g)].palette.contains(rgb) })
                }
            }
        }
        println("shapes.gif cleaned: ${out.size} bytes")
    }
}
