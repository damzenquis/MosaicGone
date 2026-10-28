package com.mosaicgone.core

import java.util.Locale

/**
 * Thread-safe stage timers for the timing breakdown in the log.
 * Top-level keys: read, pre, inference, post, mosaic, clean, write, preview, load, save, audio.
 * Sub-stages use "parent.child" (e.g. read.decode) and are reported indented under the parent.
 */
class Timing {

    /** [threads]: how many distinct threads added to it (a pool, e.g. parallel sharpening, has several) */
    class Stat(val nanos: Long, val count: Long, val thread: String, val threads: Int = 1)

    private class Acc(val thread: String) {
        var nanos = 0L
        var count = 0L
        val names = HashSet<String>()
    }

    private val stats = LinkedHashMap<String, Acc>()

    fun add(key: String, nanos: Long) {
        val name = Thread.currentThread().name
        val role = roleOf(name)
        synchronized(stats) {
            val a = stats.getOrPut(key) { Acc(role) }
            a.nanos += nanos
            a.count++
            // only pool threads count as "several threads" (the reader/writer are recreated per pass under the same role)
            if (role == "sharpen") a.names.add(name)
        }
    }

    inline fun <T> measure(key: String, block: () -> T): T {
        val t0 = System.nanoTime()
        try {
            return block()
        } finally {
            add(key, System.nanoTime() - t0)
        }
    }

    fun snapshot(): Map<String, Stat> = synchronized(stats) { stats.mapValues { Stat(it.value.nanos, it.value.count, it.value.thread, it.value.names.size) } }

    companion object {
        const val READ_THREAD = "mg-read"
        const val WRITE_THREAD = "mg-write"
        const val SHARPEN_THREAD = "mg-sharpen"

        /** read = decode/pre-processing thread, write = post-processing/encode thread, sharpen = --parallel_sharpen pool, main = AI thread */
        fun roleOf(threadName: String) = when {
            threadName.startsWith(READ_THREAD) -> "read"
            threadName.startsWith(WRITE_THREAD) -> "write"
            threadName.startsWith(SHARPEN_THREAD) -> "sharpen"
            else -> "main"
        }

        private val ORDER = listOf("load", "read", "pre", "inference", "enhance", "clean", "mosaic", "post", "sharpen_wait", "write", "preview", "audio", "save")

        fun diff(after: Map<String, Stat>, before: Map<String, Stat>): Map<String, Stat> =
            after.mapValues { (k, a) ->
                val b = before[k]
                if (b == null) a else Stat(a.nanos - b.nanos, a.count - b.count, a.thread, a.threads)
            }.filterValues { it.count > 0 }

        fun fmtDuration(sec: Double): String {
            val s = sec.toLong()
            return if (s >= 3600) "%d:%02d:%02d".format(s / 3600, s / 60 % 60, s % 60)
            else "%02d:%02d.%d".format(s / 60, s % 60, ((sec - s) * 10).toInt())
        }

        /** Table of where the time went. busy = stage time / wall time (stages on different threads overlap). */
        fun report(title: String, frames: Int, wallNanos: Long, s: Map<String, Stat>): String = buildString {
            val wall = wallNanos.coerceAtLeast(1)
            val f = frames.coerceAtLeast(1)
            append("⏱ $title: $frames frame(s) in ${fmtDuration(wall / 1e9)}")
            append(String.format(Locale.US, " (%.1f ms/frame)\n", wall / 1e6 / f))
            append(String.format(Locale.US, "   %-16s %-6s %9s %10s %10s %6s\n", "stage", "thread", "total s", "ms/frame", "ms/call", "busy"))
            val tops = s.keys.filter { '.' !in it }.sortedBy { ORDER.indexOf(it).let { i -> if (i < 0) 99 else i } }
            for (k in tops) {
                row(this, k, s.getValue(k), f, wall, top = true)
                for (c in s.keys.filter { it.startsWith("$k.") }.sorted()) row(this, c, s.getValue(c), f, wall, top = false)
            }
            // a pool's load is per thread (its busy % above is the sum over its threads)
            val poolSize = tops.groupBy { s.getValue(it).thread }.mapValues { (_, ks) -> ks.maxOf { s.getValue(it).threads }.coerceAtLeast(1) }
            val byThread = tops.groupBy { s.getValue(it).thread }.mapValues { (t, ks) -> ks.sumOf { s.getValue(it).nanos } / poolSize.getValue(t) }
            if (byThread.isNotEmpty()) {
                append("   thread load: ")
                append(
                    byThread.entries.sortedByDescending { it.value }.joinToString(" · ") { (t, n) ->
                        "$t %d%%".format(Locale.US, pct(n, wall)) + if (poolSize.getValue(t) > 1) " (each of ${poolSize.getValue(t)} threads)" else ""
                    },
                )
                val (bt, _) = byThread.maxByOrNull { it.value }!!
                val stage = tops.filter { s.getValue(it).thread == bt }.maxByOrNull { s.getValue(it).nanos }
                // the writer waiting for sharpened patches means the pool, not the AI thread, set the pace
                val wait = s["sharpen_wait"]?.let { pct(it.nanos, wall) } ?: 0
                if (wait >= 20) append(" → bottleneck: sharpening pool (the writer waited for it $wait% of the time)\n")
                else append(" → bottleneck: $bt thread ($stage)\n")
            }
        }

        private fun pct(n: Long, wall: Long) = (100.0 * n / wall).toInt()

        private fun row(sb: StringBuilder, key: String, st: Stat, frames: Int, wall: Long, top: Boolean) {
            val name = if (top) key else "  " + key.substringAfter('.')
            val busy = if (top) "%d%%".format(Locale.US, pct(st.nanos, wall)) else ""
            sb.append(
                String.format(
                    Locale.US, "   %-16s %-6s %9.1f %10.1f %10.1f %6s\n",
                    name, if (top) st.thread else "", st.nanos / 1e9, st.nanos / 1e6 / frames, st.nanos / 1e6 / st.count.coerceAtLeast(1), busy,
                ),
            )
        }
    }
}
