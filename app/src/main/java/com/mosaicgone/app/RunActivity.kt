package com.mosaicgone.app

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.MediaController
import android.widget.TextView
import android.widget.VideoView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Progress (live log and preview) and the result. */
class RunActivity : AppCompatActivity() {

    private lateinit var preview: ImageView
    private lateinit var compare: CompareView
    private lateinit var video: VideoView
    private lateinit var step: TextView
    private lateinit var bar: LinearProgressIndicator
    private lateinit var count: TextView
    private lateinit var eta: TextView
    private lateinit var saved: TextView
    private lateinit var log: TextView
    private lateinit var cancelBtn: MaterialButton
    private var shownResult: JobStatus.Done? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_run)
        preview = findViewById(R.id.preview)
        compare = findViewById(R.id.compare)
        video = findViewById(R.id.video)
        step = findViewById(R.id.step)
        bar = findViewById(R.id.bar)
        count = findViewById(R.id.count)
        eta = findViewById(R.id.eta)
        saved = findViewById(R.id.saved)
        log = findViewById(R.id.log)
        cancelBtn = findViewById(R.id.cancelBtn)

        findViewById<MaterialToolbar>(R.id.toolbar).setNavigationOnClickListener { finish() }
        cancelBtn.setOnClickListener {
            Jobs.cancelRequested = true
            cancelBtn.isEnabled = false
            cancelBtn.setText(R.string.cancelling)
        }
        findViewById<MaterialButton>(R.id.againBtn).setOnClickListener {
            Jobs.status.value = JobStatus.Idle
            finish()
        }
        findViewById<MaterialButton>(R.id.shareBtn).setOnClickListener {
            val d = shownResult ?: return@setOnClickListener
            val send = Intent(Intent.ACTION_SEND)
                .setType(resultType(d))
                .putExtra(Intent.EXTRA_STREAM, d.result)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            startActivity(Intent.createChooser(send, "Share result"))
        }
        findViewById<MaterialButton>(R.id.openBtn).setOnClickListener {
            val d = shownResult ?: return@setOnClickListener
            val view = Intent(Intent.ACTION_VIEW)
                .setDataAndType(d.result, resultType(d))
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            runCatching { startActivity(view) }
        }

        logLink = findViewById(R.id.logLink)
        logLink.setOnClickListener { logFile?.let { openLog(it) } }
        logLink.setOnLongClickListener { logFile?.let { shareLog(it) }; true }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { Jobs.status.collect { render(it) } }
                launch { Jobs.preview.collect { bmp -> if (shownResult == null && bmp != null) preview.setImageBitmap(bmp) } }
            }
        }
    }

    /** results keep the source's format (JPEG/PNG/WebP/GIF/MP4) */
    private fun resultType(d: JobStatus.Done) = contentResolver.getType(d.result) ?: if (d.isVideo) "video/mp4" else "image/*"

    private lateinit var logLink: TextView
    private var logFile: LogFile? = null

    private fun showLogLink(f: LogFile?) {
        logFile = f
        val v = if (f != null) View.VISIBLE else View.GONE
        logLink.visibility = v
        findViewById<View>(R.id.logHint).visibility = v
        if (f != null) {
            logLink.text = android.text.SpannableString("📄 ${f.name}").apply {
                setSpan(android.text.style.UnderlineSpan(), 3, length, 0)
            }
        }
    }

    private fun openLog(f: LogFile) {
        val view = Intent(Intent.ACTION_VIEW).setDataAndType(f.uri, "text/plain").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        try {
            startActivity(view)
        } catch (e: android.content.ActivityNotFoundException) {
            shareLog(f) // no text viewer installed
        }
    }

    private fun shareLog(f: LogFile) {
        val send = Intent(Intent.ACTION_SEND).setType("text/plain")
            .putExtra(Intent.EXTRA_STREAM, f.uri)
            .putExtra(Intent.EXTRA_SUBJECT, f.name)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        startActivity(Intent.createChooser(send, "Share log"))
    }

    private fun fmt(sec: Long) = if (sec >= 3600) "%d:%02d:%02d".format(sec / 3600, sec / 60 % 60, sec % 60) else "%02d:%02d".format(sec / 60, sec % 60)

    private fun render(s: JobStatus) {
        val running = s is JobStatus.Running
        cancelBtn.visibility = if (running) View.VISIBLE else View.GONE
        findViewById<View>(R.id.againBtn).visibility = if (running) View.GONE else View.VISIBLE
        findViewById<View>(R.id.metaRow).visibility = if (running) View.VISIBLE else View.GONE
        bar.visibility = if (running) View.VISIBLE else View.GONE
        when (s) {
            is JobStatus.Idle -> finish()
            is JobStatus.Running -> {
                step.text = s.step
                if (s.total > 0) {
                    bar.isIndeterminate = false
                    bar.max = s.total
                    bar.setProgressCompat(s.done, true)
                    count.text = getString(R.string.progress_count, s.done.toString(), s.total.toString())
                } else {
                    bar.isIndeterminate = true
                    count.text = ""
                }
                eta.text = if (s.etaSec >= 0) "ETA ${fmt(s.etaSec)}" else ""
                log.text = s.log.takeLast(24).joinToString("\n")
                showLogLink(null)
            }
            is JobStatus.Done -> {
                step.text = getString(R.string.finished_in, fmt(s.seconds))
                saved.visibility = View.VISIBLE
                saved.setTextColor(getColor(R.color.ok))
                saved.text = android.text.SpannableStringBuilder("✓ Saved to ${s.savedAs}").apply {
                    // saved, but part of it could not be cleaned: say so under it, in the warning colour
                    s.warning?.let { w ->
                        append('\n')
                        val start = length
                        append("⚠ ").append(w)
                        setSpan(android.text.style.ForegroundColorSpan(getColor(R.color.warn)), start, length, 0)
                    }
                }
                showLogLink(s.logFile)
                findViewById<View>(R.id.resultButtons).visibility = View.VISIBLE
                log.text = s.log.takeLast(40).joinToString("\n")
                showResult(s)
            }
            is JobStatus.Failed -> {
                step.setText(R.string.status_error)
                saved.visibility = View.VISIBLE
                saved.setTextColor(getColor(R.color.warn))
                saved.text = s.message
                showLogLink(s.logFile)
                log.text = s.log.takeLast(40).joinToString("\n")
            }
            is JobStatus.Cancelled -> {
                step.setText(R.string.status_cancelled)
                showLogLink(s.logFile)
                log.text = s.log.takeLast(40).joinToString("\n")
            }
        }
    }

    private fun showResult(d: JobStatus.Done) {
        if (shownResult == d) return
        shownResult = d
        if (d.isVideo) {
            preview.visibility = View.GONE
            video.visibility = View.VISIBLE
            video.setMediaController(MediaController(this).also { it.setAnchorView(video) })
            video.setVideoURI(d.result)
            video.setOnPreparedListener { it.isLooping = true; video.start() }
        } else {
            lifecycleScope.launch {
                val (before, after) = withContext(Dispatchers.IO) {
                    runCatching { Media.decodeBitmap(this@RunActivity, d.source, maxSide = 1440) }.getOrNull() to
                        runCatching { Media.decodeBitmap(this@RunActivity, d.result, maxSide = 1440) }.getOrNull()
                }
                if (after != null) {
                    compare.before = before
                    compare.after = after
                    preview.visibility = View.GONE
                    compare.visibility = View.VISIBLE
                }
            }
        }
    }
}
