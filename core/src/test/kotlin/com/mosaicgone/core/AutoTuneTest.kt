package com.mosaicgone.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AutoTuneTest {
    private val a17 = AutoTune.Device(perfCores = 2, effCores = 6, perfGHz = 2.4, effGHz = 2.0, heterogeneous = true, ramGB = 3.6, heapMB = 512)
    private val flagship = AutoTune.Device(perfCores = 4, effCores = 4, perfGHz = 3.3, effGHz = 2.0, heterogeneous = true, ramGB = 12.0, heapMB = 512)
    private val quad = AutoTune.Device(perfCores = 4, effCores = 4, perfGHz = 1.8, effGHz = 1.8, heterogeneous = false, ramGB = 2.0, heapMB = 256)
    private val clip = AutoTune.Media(AutoTune.Kind.VIDEO, 848, 478, fps = 25.0, durationSec = 25.4)
    private val model = Options(modelName = Options.RESTORER)

    @Test fun a17Video() {
        val r = AutoTune.choose(a17, clip, model)
        r.reasons.forEach { println(it) }
        println("estimate ${AutoTune.fmtEstimate(r.estimateSec!!)}")
        val o = r.options
        assertFalse("sharpening scored lower on the restorer", o.enhancePatch)
        assertTrue("4 spare efficiency cores: parallel sharpening ready if sharpening is turned on", o.parallelSharpen)
        // realesr-general on the A17: ~1.6 s extra per frame in parallel instead of 3.75 s on the AI thread alone
        val extra = AutoTune.parallelSharpenCost(3.75, 1.3, 4, 2.0)
        println("parallel sharpen extra %.2f s/frame".format(extra))
        assertTrue(extra in 1.2..2.0)
        assertEquals(3.75, AutoTune.parallelSharpenCost(3.75, 1.3, 0, 2.0), 0.0)
        assertEquals(Options.SR_ANIME, o.srModel)
        assertEquals(11, o.medfiltNum)
        assertEquals(0, o.minMosaicSize) // the restorer's own 32 px default
        assertEquals(128, o.maskThreshold)
        assertFalse("848x478 does not need split colour conversion", o.parallelColor)
        assertEquals(0, o.maxVideoSize)
        assertTrue(o.affinity)
        // 635 frames at 1.5 s each (restorer 1.3 + detection 0.2, as measured on the A17)
        assertTrue(r.estimateSec!! in 13 * 60.0..19 * 60.0)
        o.resolve()
        // Lada's restorer: about 2.25x the restorer's part
        val lada = AutoTune.choose(a17, clip, Options(modelName = Options.LADA))
        println("lada estimate ${AutoTune.fmtEstimate(lada.estimateSec!!)}")
        assertTrue(lada.estimateSec!! in 1.8 * r.estimateSec!!..2.2 * r.estimateSec!!)
        lada.options.resolve()
    }

    @Test fun keepsUserChoicesOutsideMoreOptions() {
        val cur = Options(modelName = Options.RESTORER, fps = 10, startTime = 5.0, lastTime = 10.0)
        val o = AutoTune.choose(a17, clip, cur).options
        assertEquals(10, o.fps); assertEquals(5.0, o.startTime, 0.0); assertEquals(10.0, o.lastTime, 0.0)
        assertEquals(Options.RESTORER, o.modelName)
        assertEquals(5, o.medfiltNum) // 0.44 s at 10 fps
    }

    @Test fun otherDevicesAndMedia() {
        val hd = AutoTune.Media(AutoTune.Kind.VIDEO, 1920, 1080, fps = 30.0, durationSec = 60.0, hdr = true, tenBit = true)
        val f = AutoTune.choose(flagship, hd, model)
        assertTrue(f.options.parallelColor)
        assertEquals(13, f.options.medfiltNum)
        // a faster phone gets a shorter estimate for the same video
        assertTrue(f.estimateSec!! < AutoTune.choose(a17, hd, model).estimateSec!!)

        val q = AutoTune.choose(quad, hd, model)
        q.reasons.forEach { println("quad: $it") }
        assertFalse(q.options.affinity)
        // 4 equal cores, all taken by the AI threads, reader and writer: no spare core for a sharpening pool
        assertFalse(q.options.parallelSharpen)

        val photo = AutoTune.choose(a17, AutoTune.Media(AutoTune.Kind.IMAGE, 4000, 3000), model)
        assertFalse(photo.options.enhancePatch || photo.options.parallelColor)
        assertEquals(null, photo.estimateSec)
    }
}
