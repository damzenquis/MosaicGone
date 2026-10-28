package com.mosaicgone.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** MosaicGone's restorer bookkeeping: crop squares, chunks, cross-fade weights, detector mask geometry. */
class RestorerPlanTest {

    private val pipe = Pipeline(Options(modelName = Options.RESTORER).resolve(), object : Progress {})

    @Test fun cropsAreSteadyAndInside() {
        // a stretch whose mosaic grows, one frame dips (detection blinked), then a gap and a second stretch near the edge
        val sizes = IntArray(60) { if (it in 5..40) 40 + it else if (it in 45..59) 30 else 0 }
        sizes[20] = 18
        val p = Array(60) { intArrayOf(if (it >= 45) 10 else 300, 200, sizes[it]) }
        val plan = pipe.planCrops(p, 640, 360, 16)
        assertEquals(listOf(5..40, 45..59), plan.stretches)
        for (r in plan.stretches) for (k in r) {
            assertTrue("frame $k on", plan.on[k])
            // never smaller than the mosaic, never outside the frame
            assertTrue("frame $k square ${plan.half[k]} < mosaic ${p[k][2]}", plan.half[k] >= p[k][2])
            assertTrue(plan.x[k] - plan.half[k] >= 0 && plan.x[k] + plan.half[k] <= 640)
            assertTrue(plan.y[k] - plan.half[k] >= 0 && plan.y[k] + plan.half[k] <= 360)
        }
        // the blink at frame 20 does not shrink the square there
        assertTrue(plan.half[20] >= plan.half[19] - 2)
        // size changes slowly inside a stretch
        for (k in 6..40) assertTrue("jump at $k", kotlin.math.abs(plan.half[k] - plan.half[k - 1]) <= 3)
        assertTrue(!plan.on[0] && !plan.on[42])
    }

    @Test fun chunksCoverEveryFrameWithPositiveWeight() {
        val stretches = listOf(0..2, 10..21, 30..79, 100..115, 200..216)
        val chunks = pipe.planChunks(stretches)
        for (r in stretches) for (f in r) {
            val cover = chunks.filter { f in it.start..it.end }
            assertTrue("frame $f not covered", cover.isNotEmpty())
            assertTrue("frame $f weight", cover.sumOf { it.weight(f).toDouble() } > 0.0)
            for (c in cover) assertTrue(c.end - c.start + 1 <= c.len)
        }
        // short stretch -> 4-frame export, 12 frames -> one 16-frame call, 50 frames -> overlapping 16s
        assertEquals(Restorer.T_SHORT, chunks.first { it.start == 0 }.len)
        assertEquals(1, chunks.count { it.start in 10..21 })
        val long = chunks.filter { it.start in 30..79 }
        assertTrue(long.all { it.len == Restorer.T && it.end - it.start + 1 == Restorer.T })
        assertTrue(long.zipWithNext().all { (a, b) -> b.start > a.start && b.start <= a.end - Restorer.OVERLAP + 1 })
        // chunks are in frame order (the writer relies on it)
        assertTrue(chunks.zipWithNext().all { (a, b) -> b.start > a.start })
        // exactly 16 and 17 frames
        assertEquals(1, chunks.count { it.start in 100..115 })
        assertEquals(2, chunks.count { it.start in 200..216 })
    }

    @Test fun crossFadeIsSmooth() {
        val c = Pipeline.ChunkPlan(10, 25, Restorer.T, rampIn = true, rampOut = true)
        assertEquals(0.2f, c.weight(10), 1e-6f)
        assertEquals(1f, c.weight(17), 1e-6f)
        assertEquals(0.2f, c.weight(25), 1e-6f)
        val edge = Pipeline.ChunkPlan(0, 15, Restorer.T, rampIn = false, rampOut = true)
        assertEquals(1f, edge.weight(0), 1e-6f)
    }

    @Test fun detectorMaskLandsWhereTheMosaicIs() {
        // a 1280x720 frame letterboxed to 416x234: the map covers 104 x 58.5 cells; mark cells for x 400..600, y 200..400
        val w = 1280; val h = 720
        val r = 416.0 / w
        val map = FloatArray(104 * 104)
        for (y in 0 until 104) for (x in 0 until 104) {
            val fx = (x + 0.5) * 4 / r; val fy = (y + 0.5) * 4 / r
            if (fx in 400.0..600.0 && fy in 200.0..400.0) map[y * 104 + x] = 1f
        }
        val seg = Nets.segFromDet(w, h, Net.Tensor(map, longArrayOf(1, 1, 104, 104)))
        assertEquals(640, seg.w); assertEquals(360, seg.h)
        val b = ImageOps.boundingRect(ImageOps.threshold(seg, 127.0))
        val s = 360.0 / h
        assertEquals(400 * s, b[0].toDouble(), 6.0)
        assertEquals(200 * s, b[1].toDouble(), 6.0)
        assertEquals(600 * s, (b[0] + b[2]).toDouble(), 6.0)
        assertEquals(400 * s, (b[1] + b[3]).toDouble(), 6.0)
    }

    @Test fun detectorInputIsLetterboxedBgr() {
        val img = Img(200, 100, 3)
        for (p in 0 until 200 * 100) { img.data[p * 3] = 10; img.data[p * 3 + 1] = 20; img.data[p * 3 + 2] = 30 }
        val t = Nets.detInput(img)
        val n = Nets.DET_SIZE * Nets.DET_SIZE
        assertEquals(30f, t.data[0], 0.01f)       // B plane first
        assertEquals(10f, t.data[2 * n], 0.01f)   // R plane last
        assertEquals(114f, t.data[(Nets.DET_SIZE - 1) * Nets.DET_SIZE], 0.01f) // padding below the 416x208 picture
    }

    @Test fun restorerModelIsRecognised() {
        val o = Options(modelName = Options.RESTORER).resolve()
        assertEquals("clean", o.mode)
        assertEquals(128, o.maskThreshold)
    }
}
