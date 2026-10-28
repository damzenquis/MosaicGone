package com.mosaicgone.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicBoolean

class StagedTest {

    private fun counter(n: Int): () -> Int? {
        var i = 0
        return { if (i < n) i++ else null }
    }

    @Test fun keepsOrderAndRunsOnThreeThreads() {
        val out = ArrayList<Int>()
        val threads = HashSet<String>()
        val c = counter(500)
        Staged(parallel = true).run<Int, Int>(
            produce = { synchronized(threads) { threads.add(Thread.currentThread().name) }; c() },
            work = { it * 2 },
            consume = { synchronized(threads) { threads.add(Thread.currentThread().name) }; out.add(it) },
            isCancelled = { false },
        )
        assertEquals((0 until 500).map { it * 2 }, out)
        assertTrue(Timing.READ_THREAD in threads && Timing.WRITE_THREAD in threads)
    }

    @Test fun sequentialModeIsInline() {
        val out = ArrayList<Int>()
        val me = Thread.currentThread().name
        Staged(parallel = false).run(counter(5), { assertEquals(me, Thread.currentThread().name); it + 1 }, { out.add(it) }, { false })
        assertEquals(listOf(1, 2, 3, 4, 5), out)
    }

    @Test fun errorsInAnyStagePropagate() {
        for (stage in 0..2) {
            val e = assertThrows(IllegalStateException::class.java) {
                val c = counter(100)
                Staged(parallel = true).run<Int, Int>(
                    produce = { val v = c(); if (stage == 0 && v == 7) error("boom$stage"); v },
                    work = { if (stage == 1 && it == 7) error("boom$stage"); it },
                    consume = { if (stage == 2 && it == 7) error("boom$stage") },
                    isCancelled = { false },
                )
            }
            assertEquals("boom$stage", e.message)
        }
    }

    @Test fun cancelStopsEverything() {
        val cancel = AtomicBoolean(false)
        var consumed = 0
        assertThrows(CancelledException::class.java) {
            Staged(parallel = true).run<Int, Int>(
                produce = { 1 },
                work = { if (consumed > 20) cancel.set(true); it },
                consume = { consumed++ },
                isCancelled = { cancel.get() },
            )
        }
    }

    @Test fun threadInitRunsOnWorkers() {
        val roles = HashSet<String>()
        Staged(parallel = true, threadInit = { synchronized(roles) { roles.add(it) } }).run(counter(3), { it }, {}, { false })
        assertEquals(setOf("read", "write"), roles)
    }

    @Test fun timingReportFormat() {
        val t = Timing()
        t.add("read", 2_000_000_000)
        t.add("read.decode", 1_500_000_000)
        t.add("inference", 9_000_000_000)
        val r = Timing.report("Step", 10, 10_000_000_000, t.snapshot())
        println(r)
        assertTrue(r.contains("1000.0 ms/frame"))
        assertTrue(r.contains("  decode"))
        assertTrue(r.contains("bottleneck: main thread (inference)"))
    }
}
