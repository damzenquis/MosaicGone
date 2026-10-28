package com.mosaicgone.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import com.sun.jna.LastErrorException
import com.sun.jna.Library
import com.sun.jna.Native
import com.sun.jna.NativeLong
import java.io.File
import java.util.Locale

/** CPU layout read from sysfs: which cores are performance (big/prime) and which are efficiency (little). */
class CpuTopology(val cores: List<Int>, val maxKHz: Map<Int, Int>, val perf: List<Int>, val eff: List<Int>, val note: String) {

    /** true when there are distinct performance cores worth pinning to */
    val heterogeneous get() = perf.isNotEmpty() && eff.isNotEmpty() && perf != eff

    fun describe(): String {
        if (maxKHz.isEmpty()) return "${cores.size} cores ($note)"
        val groups = cores.groupBy { maxKHz[it] ?: 0 }.toSortedMap()
        val parts = groups.map { (khz, cs) -> "cpu${ranges(cs)} %.2f GHz".format(Locale.US, khz / 1e6) }
        return "${cores.size} cores: " + parts.joinToString(" · ") +
            if (heterogeneous) " → performance cores: ${ranges(perf)}, efficiency cores: ${ranges(eff)}" else " ($note)"
    }

    companion object {
        fun ranges(cs: List<Int>): String {
            if (cs.isEmpty()) return "-"
            val s = cs.sorted()
            val out = ArrayList<String>()
            var a = s[0]; var b = s[0]
            for (c in s.drop(1)) {
                if (c == b + 1) b = c else { out.add(if (a == b) "$a" else "$a-$b"); a = c; b = c }
            }
            out.add(if (a == b) "$a" else "$a-$b")
            return out.joinToString(",")
        }

        private fun readInt(path: String) = runCatching { File(path).readText().trim().toInt() }.getOrNull()

        private fun parseList(s: String): List<Int> = s.trim().split(',').filter { it.isNotBlank() }.flatMap { p ->
            val r = p.split('-').map { it.trim().toInt() }
            if (r.size == 2) (r[0]..r[1]).toList() else listOf(r[0])
        }

        /** ARM "CPU part" ids of the in-order efficiency cores: Cortex-A53, A35, A55, A510, A520. */
        private val LITTLE_PARTS = setOf(0xd03, 0xd04, 0xd05, 0xd46, 0xd80)

        /** 1 for efficiency cores, 2 for everything else, per core; null when /proc/cpuinfo has no part ids. */
        private fun cpuinfoTiers(cores: List<Int>): Map<Int, Int?> {
            val parts = HashMap<Int, Int>()
            runCatching {
                var cpu = -1
                File("/proc/cpuinfo").forEachLine { line ->
                    val k = line.substringBefore(':').trim()
                    val v = line.substringAfter(':', "").trim()
                    when (k) {
                        "processor" -> cpu = v.toIntOrNull() ?: -1
                        "CPU part" -> if (cpu >= 0) v.removePrefix("0x").toIntOrNull(16)?.let { parts[cpu] = it }
                    }
                }
            }
            return cores.associateWith { c -> parts[c]?.let { if (it in LITTLE_PARTS) 1 else 2 } }
        }

        val current: CpuTopology by lazy { detect() }

        private fun detect(): CpuTopology {
            val cores = runCatching { parseList(File("/sys/devices/system/cpu/possible").readText()) }.getOrNull()
                ?: (0 until Runtime.getRuntime().availableProcessors()).toList()
            // max frequency per core; if unreadable fall back to the scheduler's cpu_capacity
            var speed = cores.associateWith { readInt("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq") }
            var unit = "freq"
            if (speed.values.any { it == null }) {
                speed = cores.associateWith { readInt("/sys/devices/system/cpu/cpu$it/cpu_capacity") }
                unit = "capacity"
            }
            if (speed.values.any { it == null }) {
                // some vendors block sysfs for apps: fall back to the ARM core type in /proc/cpuinfo
                speed = cpuinfoTiers(cores)
                unit = "/proc/cpuinfo core types"
            }
            if (speed.values.any { it == null }) return CpuTopology(cores, emptyMap(), cores, cores, "core speeds unreadable, no pinning")
            val s = speed.mapValues { it.value!! }
            val min = s.values.minOrNull()!!
            val perf = cores.filter { s.getValue(it) > min }
            val eff = cores.filter { s.getValue(it) == min }
            if (perf.isEmpty()) return CpuTopology(cores, if (unit == "freq") s else emptyMap(), cores, cores, "all cores equal, no pinning")
            return CpuTopology(cores, if (unit == "freq") s else emptyMap(), perf, eff, "from $unit")
        }
    }
}

/** sched_setaffinity / sched_getaffinity from bionic libc via JNA (no NDK needed). */
object Affinity {
    @Suppress("FunctionName")
    private interface LibC : Library {
        @Throws(LastErrorException::class)
        fun sched_setaffinity(pid: Int, cpusetsize: NativeLong, mask: ByteArray): Int

        @Throws(LastErrorException::class)
        fun sched_getaffinity(pid: Int, cpusetsize: NativeLong, mask: ByteArray): Int
    }

    private const val SET_BYTES = 128 // sizeof(cpu_set_t): 1024 bits

    private val libc: LibC? by lazy { runCatching { Native.load("c", LibC::class.java) }.getOrNull() }

    val available get() = libc != null

    /** Pins thread [tid] (0 = calling thread). Returns null on success, else the reason. */
    fun set(tid: Int, cpus: List<Int>): String? {
        val lib = libc ?: return "libc not loadable"
        val mask = ByteArray(SET_BYTES)
        for (c in cpus) if (c in 0 until SET_BYTES * 8) mask[c / 8] = (mask[c / 8].toInt() or (1 shl (c % 8))).toByte()
        return try {
            lib.sched_setaffinity(tid, NativeLong(SET_BYTES.toLong()), mask)
            null
        } catch (e: LastErrorException) {
            when (e.errorCode) {
                22 -> "EINVAL (cores not allowed for this app right now)"
                3 -> "ESRCH (thread gone)"
                1 -> "EPERM"
                else -> "errno ${e.errorCode}"
            }
        } catch (e: Throwable) {
            e.toString()
        }
    }

    fun get(tid: Int): List<Int>? {
        val lib = libc ?: return null
        val mask = ByteArray(SET_BYTES)
        return try {
            lib.sched_getaffinity(tid, NativeLong(SET_BYTES.toLong()), mask)
            (0 until SET_BYTES * 8).filter { mask[it / 8].toInt() and (1 shl (it % 8)) != 0 }
        } catch (e: Throwable) {
            null
        }
    }

    /** All thread ids of this process. */
    fun threadIds(): Set<Int> = File("/proc/self/task").list()?.mapNotNull { it.toIntOrNull() }?.toSet() ?: emptySet()
}

/**
 * Per-job core binding: AI threads (the job thread and ONNX Runtime's pool) -> performance cores,
 * image-handling threads (reader/writer stages) -> efficiency cores, so they do not steal time from the AI.
 * Android can reset a thread's affinity when the app changes scheduling group (e.g. screen off), so
 * [heartbeat] re-checks and re-applies it.
 */
class CoreBinder(enabled: Boolean, private val log: (String) -> Unit) {
    val topo = CpuTopology.current
    val active = enabled && topo.heterogeneous && Affinity.available
    private val ai = LinkedHashSet<Int>()
    private val io = LinkedHashSet<Int>()
    private var resets = 0
    private var failures = 0

    val status: String = when {
        !enabled -> "off (--no_affinity)"
        !topo.heterogeneous -> "not needed (${topo.note})"
        !Affinity.available -> "unavailable (libc not loadable)"
        else -> "AI → cpu ${CpuTopology.ranges(topo.perf)}, image handling → cpu ${CpuTopology.ranges(topo.eff)}"
    }

    @Synchronized
    private fun bind(tid: Int, cpus: List<Int>, set: MutableSet<Int>, what: String) {
        Affinity.set(tid, cpus)?.let { err ->
            failures++
            if (failures <= 3) log("   affinity: could not pin $what (tid $tid): $err")
            return
        }
        set.add(tid)
    }

    /** Call on the AI thread before loading models: ORT's pool threads inherit this mask when created. */
    fun bindCurrentAsAi() {
        if (active) bind(android.os.Process.myTid(), topo.perf, ai, "AI thread")
    }

    /** Threads created while a model was loading are ONNX Runtime's intra-op pool. */
    fun adoptOrtThreads(before: Set<Int>) {
        if (!active) return
        val created = Affinity.threadIds() - before
        for (tid in created) bind(tid, topo.perf, ai, "ORT thread")
        if (created.isNotEmpty()) log("   affinity: ${created.size} ONNX Runtime thread(s) pinned to cpu ${CpuTopology.ranges(topo.perf)}")
    }

    fun bindCurrentAsIo() {
        if (active) bind(android.os.Process.myTid(), topo.eff, io, "image thread")
    }

    private val resetsPerThread = HashMap<Int, Int>()
    private var warnedBackground = false
    private var released = 0

    /**
     * Re-applies the masks if the system reset them; forgets threads that have ended.
     * A thread Android keeps moving back is left to the system after [MAX_REPINS] re-pins instead of fighting it
     * every heartbeat (each fight costs a syscall and a migration, and the system wins again anyway).
     */
    @Synchronized
    fun heartbeat() {
        if (!active) return
        for ((set, cpus) in listOf(ai to topo.perf, io to topo.eff)) {
            val it = set.iterator()
            while (it.hasNext()) {
                val tid = it.next()
                val now = Affinity.get(tid)
                if (now == null) { it.remove(); continue }
                if (now.sorted() == cpus.sorted()) continue
                val n = (resetsPerThread[tid] ?: 0) + 1
                if (n > MAX_REPINS) {
                    it.remove()
                    released++
                    log("   affinity: Android keeps moving tid $tid to cpu ${CpuTopology.ranges(now)}; left to the system after $MAX_REPINS re-pins")
                    if (now.none { c -> c in topo.perf } && !warnedBackground) {
                        warnedBackground = true
                        log("   note: Android limited this app to the efficiency cores, usually because another app is in use; the job runs slower until it is the main app again")
                    }
                    continue
                }
                resetsPerThread[tid] = n
                if (Affinity.set(tid, cpus) == null) {
                    resets++
                    if (resets <= 3) log("   affinity: system reset tid $tid to cpu ${CpuTopology.ranges(now)}; re-pinned (#$resets)")
                } else it.remove()
            }
        }
    }

    fun summary(): String = "affinity: $status · pinned AI threads ${ai.size}, image threads ${io.size}, re-pins $resets" +
        (if (released > 0) ", left to the system $released" else "") + ", failures $failures"

    private companion object {
        const val MAX_REPINS = 3
    }
}

/** Thermal / clock / battery state, logged at each step and every 10% so throttling shows up in the log. */
object DeviceStatus {
    fun describe(ctx: Context, topo: CpuTopology = CpuTopology.current): String {
        val parts = ArrayList<String>()
        val pm = ctx.getSystemService(PowerManager::class.java)
        val thermal = when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "none"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "severe"
            PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
            PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
            else -> "?"
        }
        var t = "thermal $thermal"
        if (Build.VERSION.SDK_INT >= 30) {
            val hr = runCatching { pm.getThermalHeadroom(0) }.getOrDefault(Float.NaN)
            if (!hr.isNaN()) t += " (headroom %.2f)".format(Locale.US, hr)
        }
        parts.add(t)
        val probe = topo.perf.firstOrNull() ?: topo.cores.firstOrNull()
        if (probe != null) {
            val cur = readKHz("/sys/devices/system/cpu/cpu$probe/cpufreq/scaling_cur_freq")
            val cap = readKHz("/sys/devices/system/cpu/cpu$probe/cpufreq/scaling_max_freq")
            if (cur != null) parts.add("cpu$probe %.2f GHz".format(Locale.US, cur / 1e6) + (cap?.let { " (cap %.2f)".format(Locale.US, it / 1e6) } ?: ""))
        }
        ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))?.let { b ->
            val temp = b.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            val level = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100)
            val plugged = b.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0
            var s = "battery"
            if (temp != Int.MIN_VALUE) s += " %.1f °C".format(Locale.US, temp / 10.0)
            if (level >= 0) s += " ${level * 100 / scale.coerceAtLeast(1)}%"
            s += if (plugged) " charging" else " on battery"
            parts.add(s)
        }
        parts.add("power saving " + if (pm.isPowerSaveMode) "ON" else "off")
        return parts.joinToString(" · ")
    }

    private fun readKHz(path: String) = runCatching { File(path).readText().trim().toInt() }.getOrNull()
}
