package com.mosaicgone.core

import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * --parallel_color: splits per-row image work (YUV <-> RGB) over [threads] helper threads plus the calling thread.
 * Shared by the reader and writer stages; each call waits for its own parts only. Results are identical to one thread.
 */
class RowPool(val threads: Int, threadInit: () -> Unit = {}) : AutoCloseable {
    private val n = AtomicInteger()
    private val ex: ExecutorService = Executors.newFixedThreadPool(threads) { r ->
        Thread({ threadInit(); r.run() }, "$THREAD-${n.incrementAndGet()}").apply { isDaemon = true }
    }

    /** Calls [body] on row ranges covering 0 until [rows]: one on the calling thread, the rest in the pool. */
    fun rows(rows: Int, body: (from: Int, to: Int) -> Unit) {
        val parts = minOf(threads + 1, rows / MIN_ROWS).coerceAtLeast(1)
        if (parts == 1) { body(0, rows); return }
        val step = (rows + parts - 1) / parts
        val futures = ArrayList<Future<*>>(parts - 1)
        for (p in 1 until parts) {
            val a = p * step
            val b = minOf(rows, a + step)
            if (a < b) futures.add(ex.submit { body(a, b) })
        }
        body(0, minOf(rows, step))
        for (f in futures) try {
            f.get()
        } catch (e: ExecutionException) {
            throw e.cause ?: e
        }
    }

    override fun close() {
        ex.shutdownNow()
        ex.awaitTermination(10, TimeUnit.SECONDS)
    }

    companion object {
        const val THREAD = "mg-color"
        /** smaller bands are not worth a thread hand-off */
        const val MIN_ROWS = 16

        fun run(pool: RowPool?, rows: Int, body: (from: Int, to: Int) -> Unit) {
            if (pool == null) body(0, rows) else pool.rows(rows, body)
        }
    }
}
