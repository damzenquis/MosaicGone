package com.mosaicgone.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread

class JobGuardTest {

    @Test
    fun everyLetsOneCallerThroughPerInterval() {
        val every = JobGuard.Every(5000)
        assertTrue("the first call is due", every.due(10_000))
        assertFalse(every.due(12_000))
        assertFalse(every.due(15_000))
        assertTrue(every.due(15_001))
    }

    /** threads arriving at the same moment: exactly one of them writes the log copy */
    @Test
    fun everyIsOneWinnerUnderContention() {
        repeat(200) {
            val every = JobGuard.Every(5000)
            val start = CountDownLatch(1)
            val wins = AtomicInteger()
            val threads = (1..8).map { thread { start.await(); if (every.due(100_000)) wins.incrementAndGet() } }
            start.countDown()
            threads.forEach { it.join() }
            assertEquals(1, wins.get())
        }
    }
}
