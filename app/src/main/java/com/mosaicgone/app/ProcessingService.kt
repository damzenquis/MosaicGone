package com.mosaicgone.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.net.Uri
import com.mosaicgone.core.CancelledException
import com.mosaicgone.core.Timing
import kotlin.concurrent.thread

/** Keeps a job running while the screen is off or another app is open. */
class ProcessingService : Service() {

    enum class StartAction { NOTHING_TO_RUN, ALREADY_RUNNING, START_JOB }

    private var wakeLock: PowerManager.WakeLock? = null
    private var lastNotify = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        JobGuard.installCrashLog(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val req = Jobs.request
        val action = startAction(req != null, jobActive)
        if (action == StartAction.NOTHING_TO_RUN) {
            // restarted by the system after the app's process ended: nothing to resume, clear the old notification
            getSystemService(NotificationManager::class.java).cancel(NOTIFY_ID)
            stopSelf(startId)
            return START_NOT_STICKY
        }
        // every startForegroundService() must be answered with startForeground(), or Android ends the app
        createChannel()
        val type = if (Build.VERSION.SDK_INT >= 35) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        else ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        if (action == StartAction.ALREADY_RUNNING) {
            startForeground(NOTIFY_ID, notification("Processing…", 0, 0, ongoing = true), type)
            return START_NOT_STICKY
        }
        startForeground(NOTIFY_ID, notification("Starting…", 0, 0, ongoing = true), type)
        wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MosaicGone:job").apply { acquire(6 * 3600 * 1000L) }

        jobActive = true
        thread(name = "mosaicgone-job") { runJob(req!!, startId) }
        return START_NOT_STICKY
    }

    private fun runJob(req: JobRequest, startId: Int) {
        val t0 = SystemClock.elapsedRealtime()
        val timing = Timing()
        JobGuard.jobStarted(applicationContext, req.displayName)
        // progress updates arrive from several threads: only one of them writes the log copy per 5 s
        val logCopyDue = JobGuard.Every(5000)
        lateinit var progress: JobProgress
        progress = JobProgress(applicationContext, timing) { st ->
            Jobs.status.value = st
            val now = SystemClock.elapsedRealtime()
            if (now - lastNotify > 1000) {
                lastNotify = now
                notify(notification(st.step, st.done, st.total, ongoing = true))
            }
            // if Android ends the process, the next app start can still show what happened
            if (logCopyDue.due(now)) JobGuard.saveLog(applicationContext, progress.text())
        }
        var logName = "${req.displayName.substringBeforeLast('.').ifEmpty { "job" }}_log.txt"
        var warning: String? = null
        val result: Result<Pair<Uri, String>> = try {
            val runner = JobRunner(applicationContext, req, progress, timing)
            logName = runner.logName
            Result.success(runner.run()).also { warning = runner.warning }
        } catch (e: Throwable) {
            Result.failure(e)
        }
        val error = result.exceptionOrNull()
        when (error) {
            null -> progress.log("Finished! Saved to ${result.getOrThrow().second}" + (warning?.let { "\nWarning: $it" } ?: ""))
            is CancelledException -> progress.log("Cancelled.")
            else -> progress.log("ERROR: $error\n" + error.stackTraceToString().lines().take(12).joinToString("\n"))
        }
        // the full log as a .txt next to the user's downloads, linked from the result screen
        val logFile = try {
            LogFile(Media.saveText(applicationContext, progress.text(), logName), logName)
        } catch (e: Exception) {
            progress.log("Could not save the log file: ${e.message}")
            null
        }
        val final: JobStatus = when (error) {
            null -> result.getOrThrow().let { (uri, savedAs) ->
                JobStatus.Done(uri, req.isVideo, req.media, savedAs, (SystemClock.elapsedRealtime() - t0) / 1000, progress.log, logFile, warning)
            }
            is CancelledException -> JobStatus.Cancelled(progress.log, logFile)
            is OutOfMemoryError -> JobStatus.Failed(
                "Not enough memory. Try a smaller video size (More Options → max video size) or a smaller photo.", progress.log, logFile,
            )
            else -> JobStatus.Failed(error.message ?: error.toString(), progress.log, logFile)
        }
        JobGuard.jobEnded(applicationContext)
        stopForeground(STOP_FOREGROUND_DETACH)
        notify(
            notification(
                when (final) {
                    is JobStatus.Done -> if (final.warning == null) "Finished! Saved to ${final.savedAs}"
                    else "Finished with a warning: saved to ${final.savedAs}"
                    is JobStatus.Cancelled -> "Cancelled"
                    is JobStatus.Failed -> "Error: ${final.message}"
                    else -> ""
                },
                0, 0, ongoing = false,
            ),
        )
        wakeLock?.let { if (it.isHeld) it.release() }
        // the service is done with this job before the screens hear of it: a job started from the result screen
        // then finds it free (and this stop only applies if no newer start came in)
        jobActive = false
        stopSelf(startId)
        Jobs.status.value = final
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        // Android 15 limits background media processing time; stop cleanly instead of being killed
        Jobs.cancelRequested = true
    }

    override fun onDestroy() {
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    private fun createChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Processing", NotificationManager.IMPORTANCE_LOW))
    }

    private fun notify(n: Notification) = getSystemService(NotificationManager::class.java).notify(NOTIFY_ID, n)

    private fun notification(text: String, done: Int, total: Int, ongoing: Boolean): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, RunActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_mosaic)
            .setContentTitle("MosaicGone")
            .setContentText(if (total > 0) "$text  $done/$total" else text)
            .setContentIntent(open)
            .setOngoing(ongoing)
            .setAutoCancel(!ongoing)
            .setOnlyAlertOnce(true)
            .apply { if (ongoing) setProgress(total, done, total == 0) }
            .build()
    }

    companion object {
        private const val CHANNEL = "jobs"
        private const val NOTIFY_ID = 1
        /** a job's thread is running and has not finished its wind-down (notification, wake lock, service stop) */
        @Volatile private var jobActive = false

        /** What a start request does: run [Jobs.request], or (no request, or a job still running) only acknowledge it. */
        fun startAction(hasRequest: Boolean, jobActive: Boolean) = when {
            !hasRequest -> StartAction.NOTHING_TO_RUN
            jobActive -> StartAction.ALREADY_RUNNING
            else -> StartAction.START_JOB
        }

        fun start(ctx: Context, req: JobRequest) {
            Jobs.request = req
            Jobs.cancelRequested = false
            Jobs.preview.value = null
            Jobs.status.value = JobStatus.Running("Starting", 0, 0, -1, emptyList())
            ctx.startForegroundService(Intent(ctx, ProcessingService::class.java))
        }
    }
}
