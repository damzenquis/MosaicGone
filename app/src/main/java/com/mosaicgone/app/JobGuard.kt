package com.mosaicgone.app

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * A job dies with the app's process if Android ends it (out of memory, battery saving, swiped away and stopped by the
 * phone's maker). Nothing survives that, so the job leaves a marker and a running copy of its log; the next app start
 * finds them, asks Android why the process ended (ApplicationExitInfo) and tells the user, with the log saved.
 */
object JobGuard {
    private const val PREFS = "job_guard"
    private const val KEY_JOB = "running_job"
    private const val KEY_STARTED = "started_at"
    private const val NOTIFY_ID = 1

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private fun logCopy(ctx: Context) = File(ctx.filesDir, "running_job_log.txt")
    private fun crashFile(ctx: Context) = File(ctx.filesDir, "last_crash.txt")

    // lint (ApplySharedPref) prefers apply(); commit() is deliberate in jobStarted / jobEnded: the marker must be on disk
    // before Android can end the process, or the next start cannot tell that a job was lost
    @SuppressLint("ApplySharedPref")
    fun jobStarted(ctx: Context, name: String) {
        prefs(ctx).edit().putString(KEY_JOB, name).putLong(KEY_STARTED, System.currentTimeMillis()).commit()
        logCopy(ctx).delete()
    }

    /**
     * The log so far, kept on disk while the job runs (written every few seconds from the job's progress updates, which
     * come from several threads): one writer at a time, so the copy is never two writes mixed together.
     */
    @Synchronized
    fun saveLog(ctx: Context, text: String) {
        runCatching { logCopy(ctx).writeText(text) }
    }

    @SuppressLint("ApplySharedPref") // see jobStarted
    fun jobEnded(ctx: Context) {
        prefs(ctx).edit().remove(KEY_JOB).remove(KEY_STARTED).commit()
        logCopy(ctx).delete()
    }

    /** Java crashes on any thread: the stack goes to a file first (ApplicationExitInfo only keeps a short line). */
    fun installCrashLog(ctx: Context) {
        val app = ctx.applicationContext
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        if (prev is Handler) return
        Thread.setDefaultUncaughtExceptionHandler(Handler(app, prev))
    }

    private class Handler(val ctx: Context, val prev: Thread.UncaughtExceptionHandler?) : Thread.UncaughtExceptionHandler {
        override fun uncaughtException(t: Thread, e: Throwable) {
            runCatching { crashFile(ctx).writeText("Crash on thread ${t.name}:\n${e.stackTraceToString()}") }
            prev?.uncaughtException(t, e)
        }
    }

    /**
     * At most once per [ms], whichever thread asks first: [due] is true for exactly one caller per interval (a
     * compare-and-set, so threads arriving together cannot both win).
     */
    class Every(private val ms: Long) {
        private val last = java.util.concurrent.atomic.AtomicLong(Long.MIN_VALUE / 2)

        fun due(now: Long): Boolean {
            val prev = last.get()
            return now - prev > ms && last.compareAndSet(prev, now)
        }
    }

    class Interrupted(val job: String, val reason: String, val savedLog: String?)

    /**
     * Call when the app starts with no job running in this process: a job that was running when the process ended
     * left its marker. Returns what happened (and saves its log to Download/MosaicGone), or null.
     */
    fun checkInterrupted(ctx: Context): Interrupted? {
        val p = prefs(ctx)
        val job = p.getString(KEY_JOB, null) ?: return null
        // the job's notification may still be showing although nothing runs any more
        ctx.getSystemService(NotificationManager::class.java)?.cancel(NOTIFY_ID)
        val started = p.getLong(KEY_STARTED, 0L)
        val reason = exitReason(ctx, started)
        val crash = crashFile(ctx).takeIf { it.exists() }?.readText()
        val log = buildString {
            append(logCopy(ctx).takeIf { it.exists() }?.readText() ?: "(no log was written before the app was stopped)\n")
            append("\n--- The app was stopped while this job ran.\n")
            append("Reported by Android: $reason\n")
            if (crash != null) append("\n$crash\n")
        }
        val name = job.substringBeforeLast('.').ifEmpty { "job" } + "_stopped_log.txt"
        val saved = runCatching { Media.saveText(ctx, log, name); "Download/MosaicGone/$name" }.getOrNull()
        crashFile(ctx).delete()
        jobEnded(ctx)
        return Interrupted(job, reason, saved)
    }

    private fun exitReason(ctx: Context, since: Long): String {
        if (Build.VERSION.SDK_INT < 30) return "unknown (Android 11 or newer reports this)"
        val am = ctx.getSystemService(ActivityManager::class.java) ?: return "unknown"
        val info = runCatching { am.getHistoricalProcessExitReasons(ctx.packageName, 0, 5) }.getOrNull()
            ?.firstOrNull { it.timestamp >= since } ?: return "unknown (Android kept no record)"
        val what = when (info.reason) {
            ApplicationExitInfo.REASON_LOW_MEMORY -> "the phone ran out of memory and closed the app"
            ApplicationExitInfo.REASON_USER_REQUESTED -> "the app was force-stopped (Settings, or swiped away on some phones)"
            ApplicationExitInfo.REASON_USER_STOPPED -> "the app was stopped by the user"
            ApplicationExitInfo.REASON_CRASH, ApplicationExitInfo.REASON_CRASH_NATIVE -> "the app crashed"
            ApplicationExitInfo.REASON_ANR -> "the app stopped responding"
            ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "Android stopped it for using too much battery or CPU in the background"
            ApplicationExitInfo.REASON_FREEZER -> "Android froze and then closed the app in the background"
            ApplicationExitInfo.REASON_SIGNALED -> "the system ended the process"
            ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "a permission was changed"
            ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "a system component it needed ended"
            else -> "the system closed the app"
        }
        val time = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date(info.timestamp))
        val desc = info.description?.takeIf { it.isNotBlank() }?.let { " ($it)" } ?: ""
        val mem = if (info.pss > 0) String.format(Locale.US, ", app memory then %.0f MB", info.pss / 1024.0) else ""
        return "$what at $time$desc [reason ${info.reason}, status ${info.status}$mem]"
    }
}
