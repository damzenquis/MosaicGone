package com.mosaicgone.core

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * Ordered three-stage pipeline so image handling overlaps the AI:
 *   [produce] (decode + pre-processing) on a reader thread,
 *   [work]    (inference) on the calling thread,
 *   [consume] (post-processing + encoding) on a writer thread.
 * Items stay in order; at most [depth] items wait between stages. The first failure in any stage
 * stops all three and is rethrown on the calling thread.
 * With parallel = false everything runs inline on the calling thread (A/B timing and tests).
 */
class Staged(private val parallel: Boolean, private val threadInit: (String) -> Unit = {}, private val depth: Int = 2) {

    private object End

    fun <P : Any, R : Any> run(produce: () -> P?, work: (P) -> R, consume: (R) -> Unit, isCancelled: () -> Boolean) {
        if (!parallel) {
            while (true) {
                if (isCancelled()) throw CancelledException()
                val p = produce() ?: break
                consume(work(p))
            }
            return
        }
        val inQ = ArrayBlockingQueue<Any>(depth)
        val outQ = ArrayBlockingQueue<Any>(depth)
        val stop = AtomicBoolean(false)
        val error = AtomicReference<Throwable?>(null)

        fun fail(t: Throwable) {
            error.compareAndSet(null, t)
            stop.set(true)
        }

        fun put(q: ArrayBlockingQueue<Any>, v: Any): Boolean {
            while (!q.offer(v, 50, TimeUnit.MILLISECONDS)) if (stop.get()) return false
            return true
        }

        val reader = thread(name = Timing.READ_THREAD) {
            try {
                threadInit("read")
                while (!stop.get()) {
                    val p = produce() ?: break
                    if (!put(inQ, p)) return@thread
                }
                put(inQ, End)
            } catch (t: Throwable) {
                fail(t)
            }
        }
        val writer = thread(name = Timing.WRITE_THREAD) {
            try {
                threadInit("write")
                while (true) {
                    val r = outQ.poll(50, TimeUnit.MILLISECONDS)
                    if (r == null) {
                        if (stop.get()) return@thread
                        continue
                    }
                    if (r === End || stop.get()) return@thread
                    @Suppress("UNCHECKED_CAST")
                    consume(r as R)
                }
            } catch (t: Throwable) {
                fail(t)
            }
        }
        try {
            while (true) {
                error.get()?.let { throw it }
                if (isCancelled()) throw CancelledException()
                val p = inQ.poll(50, TimeUnit.MILLISECONDS) ?: continue
                if (p === End) break
                @Suppress("UNCHECKED_CAST")
                val r = work(p as P)
                if (!put(outQ, r)) break
            }
            put(outQ, End)
            writer.join() // let the writer finish everything already queued
            error.get()?.let { throw it }
        } catch (t: Throwable) {
            fail(t)
            throw t
        } finally {
            stop.set(true)
            reader.join()
            writer.join()
        }
    }
}
