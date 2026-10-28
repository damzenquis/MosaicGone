package com.mosaicgone.app

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextWatcher
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.util.Size
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.mosaicgone.core.AutoTune
import com.mosaicgone.core.Gif
import com.mosaicgone.core.OptionError
import com.mosaicgone.core.Options
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.MaterialAutoCompleteTextView
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private var mediaUri: Uri? = null
    /** chosen, its type and name still being read (see [setMedia]) */
    private var loadingUri: Uri? = null
    private var mediaIsVideo = false
    private var mediaName = ""
    private var modelName: String? = null

    private lateinit var mediaText: TextView
    private lateinit var mediaThumb: ImageView
    private lateinit var modelText: TextView
    private lateinit var mode: MaterialAutoCompleteTextView
    private lateinit var fps: TextInputEditText
    private lateinit var hint: TextView
    private lateinit var cmd: TextView

    // More options (same names as the command-line flags)
    private lateinit var oMaskThreshold: TextInputEditText
    private lateinit var oEnhancePatch: MaterialCheckBox
    private lateinit var oSrModel: MaterialAutoCompleteTextView
    private lateinit var oDenoise: MaterialAutoCompleteTextView
    private lateinit var oMedfilt: TextInputEditText
    private lateinit var oMinMosaic: TextInputEditText
    private lateinit var oExMult: TextInputEditText
    private lateinit var oNoFeather: MaterialCheckBox
    private lateinit var oAllArea: MaterialCheckBox
    private lateinit var oStart: TextInputEditText
    private lateinit var oDuration: TextInputEditText
    private lateinit var oMaxVideo: TextInputEditText
    private lateinit var oThreads: TextInputEditText
    private lateinit var oAffinity: MaterialCheckBox
    private lateinit var oParallelSharpen: MaterialCheckBox
    private lateinit var oParallelColor: MaterialCheckBox
    private lateinit var oExtra: TextInputEditText

    private val prefs by lazy { getSharedPreferences("options", MODE_PRIVATE) }

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) setMedia(uri)
    }

    /** any file from any folder or app (the gallery picker only lists what the media library indexed) */
    private val openFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            // keeps the job's access to the file even if the app is restarted while it runs
            runCatching { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            setMedia(uri)
        }
    }

    private fun chooseMedia() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Choose an image or video")
            .setItems(
                arrayOf(
                    twoLine("Gallery", "Photos and videos on this phone"),
                    twoLine("Files", "Any folder, SD card, USB drive or cloud app"),
                ),
            ) { _, which ->
                if (which == 0) pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo))
                else openFile.launch(arrayOf("image/*", "video/*", "application/octet-stream"))
            }
            .show()
    }

    private val askNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) { checkBatteryThenRun() }

    /** back from Android's "let the app run in the background?" dialog, whatever was answered */
    private val askBattery = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { startRun() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        mediaText = findViewById(R.id.mediaText)
        mediaThumb = findViewById(R.id.mediaThumb)
        modelText = findViewById(R.id.modelText)
        mode = findViewById(R.id.mode)
        fps = findViewById(R.id.fps)
        hint = findViewById(R.id.hint)

        findViewById<TextView>(R.id.themeBtn).apply {
            val dark = Theme.isDark(this@MainActivity)
            setText(if (dark) R.string.theme_dark else R.string.theme_light)
            setOnClickListener { Theme.set(this@MainActivity, !dark) }
        }
        Links.openOnTap(modelText)

        mode.setSimpleItems(MODES.toTypedArray())
        buildMoreOptions(findViewById(R.id.morePanel))
        restorePrefs()
        // switching the theme rebuilds this screen: put back what was chosen and typed
        savedInstanceState?.let { restoreForm(it) }

        findViewById<MaterialCardView>(R.id.cardMedia).setOnClickListener { chooseMedia() }
        findViewById<MaterialCardView>(R.id.cardModel).setOnClickListener { chooseModel() }
        findViewById<MaterialCheckBox>(R.id.moreCheck).setOnCheckedChangeListener { _, on ->
            findViewById<View>(R.id.morePanel).visibility = if (on) View.VISIBLE else View.GONE
        }
        findViewById<MaterialButton>(R.id.runBtn).setOnClickListener { onRun() }
        findViewById<MaterialButton>(R.id.autoBtn).setOnClickListener { autoSelect() }
        findViewById<MaterialButton>(R.id.supportBtn).setOnClickListener { showSupport() }
        findViewById<MaterialButton>(R.id.helpBtn).setOnClickListener { showHelp() }
        findViewById<MaterialButton>(R.id.aboutBtn).setOnClickListener { showAbout() }
        findViewById<MaterialButton>(R.id.creditsBtn).setOnClickListener { showCredits() }
        findViewById<MaterialButton>(R.id.licenseBtn).setOnClickListener { showLicense() }
        mode.addTextChangedListener(watcher)
        fps.addTextChangedListener(watcher)
        refresh()
        JobGuard.installCrashLog(this)
        // a fresh process with no job: if one was running when Android ended the app, say so (with its log)
        if (!Jobs.isBusy) lifecycleScope.launch {
            val stopped = withContext(Dispatchers.IO) { runCatching { JobGuard.checkInterrupted(applicationContext) }.getOrNull() }
            if (stopped != null) showInterrupted(stopped)
        }
    }

    private fun showInterrupted(s: JobGuard.Interrupted) {
        MaterialAlertDialogBuilder(this)
            .setTitle("The last job was stopped")
            .setMessage(
                "Android closed MosaicGone while it was cleaning ${s.job}, so that job was lost.\n\n" +
                    "Reason: ${s.reason}\n\n" +
                    (if (s.savedLog != null) "Its log was saved to ${s.savedLog}.\n\n" else "") +
                    "To avoid this, allow MosaicGone to run in the background: Settings → Apps → MosaicGone → Battery → Unrestricted.",
            )
            .setPositiveButton("OK", null)
            .setNeutralButton("Battery settings") { _, _ ->
                runCatching {
                    startActivity(Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName")))
                }
            }
            .show()
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    override fun onSaveInstanceState(out: Bundle) {
        super.onSaveInstanceState(out)
        out.putParcelable("media", mediaUri ?: loadingUri)
        out.putString("model", modelName)
        out.putString("sr_model", oSrModel.text.toString())
        out.putString("denoise", oDenoise.text.toString())
        for ((k, et) in textFields()) out.putString(k, et.text.toString())
        for ((k, cb) in checkBoxes()) out.putBoolean(k, cb.isChecked)
    }

    private fun restoreForm(b: Bundle) {
        @Suppress("DEPRECATION")
        (b.getParcelable("media") as Uri?)?.let { setMedia(it) }
        modelName = b.getString("model")
        b.getString("sr_model")?.let { oSrModel.setText(it, false) }
        b.getString("denoise")?.let { oDenoise.setText(it, false) }
        for ((k, et) in textFields()) b.getString(k)?.let { et.setText(it) }
        for ((k, cb) in checkBoxes()) cb.isChecked = b.getBoolean(k, cb.isChecked)
        findViewById<View>(R.id.morePanel).visibility =
            if (findViewById<MaterialCheckBox>(R.id.moreCheck).isChecked) View.VISIBLE else View.GONE
        showModel()
        refresh()
    }

    // ------------------------------------------------------------------ Step 1 / Step 2

    /**
     * The file's type and name come from the app that provides it (a cloud app may even download it to answer), so they
     * are read off the main thread; until they arrive [mediaUri] stays null and Run / Auto-Select wait.
     */
    private fun setMedia(uri: Uri) {
        mediaUri = null
        mediaName = ""
        loadingUri = uri
        mediaText.setText(R.string.reading_file)
        mediaText.setTextColor(getColor(R.color.placeholder))
        mediaThumb.visibility = View.VISIBLE
        mediaThumb.setImageDrawable(null)
        refresh()
        lifecycleScope.launch {
            val info = withContext(Dispatchers.IO) {
                runCatching { Media.isVideo(this@MainActivity, uri) to Media.displayName(this@MainActivity, uri) }
            }
            if (loadingUri != uri) return@launch // another file was chosen meanwhile
            loadingUri = null
            info.onFailure {
                mediaThumb.visibility = View.GONE
                mediaText.setText(R.string.pick_media)
                refresh()
                hint.text = getString(R.string.cannot_read_file, it.message ?: it.javaClass.simpleName)
                return@launch
            }
            val (video, name) = info.getOrThrow()
            mediaUri = uri
            mediaIsVideo = video
            mediaName = name
            mediaText.text = twoLine(name, if (video) "Video" else "Image")
            mediaText.setTextColor(getColor(R.color.ink))
            refresh()
            val bmp = withContext(Dispatchers.IO) {
                runCatching { contentResolver.loadThumbnail(uri, Size(144, 144), null) }.getOrNull()
            }
            if (uri == mediaUri) mediaThumb.setImageBitmap(bmp)
        }
    }

    /** what to say when Run / Auto-Select need a file that is not there (yet) */
    private fun noMediaHint() = if (loadingUri != null) getString(R.string.reading_file) else "Choose an image or video first"

    // lint (PrivateResource): mtrl_alert_select_dialog_item is Material's own list row, used so the rows look exactly like
    // setItems() while a link in a description stays tappable; Material is pinned (1.12.0), so recheck this on an upgrade
    @SuppressLint("PrivateResource")
    private fun chooseModel() {
        val items = ModelRegistry.models.map { twoLine(it.title, it.description) }
        // title plus one line under it (a dialog with a list has no room for a message)
        val title = TextView(this).apply {
            text = SpannableStringBuilder("Choose Pre-trained Model:").also { sb ->
                sb.setSpan(RelativeSizeSpan(1.25f), 0, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                sb.setSpan(android.text.style.StyleSpan(Typeface.BOLD), 0, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                sb.append('\n')
                val start = sb.length
                sb.append("Both models use a YOLOX-Nano for detecting mosaics.")
                sb.setSpan(ForegroundColorSpan(getColor(R.color.muted)), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                sb.setSpan(RelativeSizeSpan(0.9f), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            }
            setTextColor(getColor(R.color.ink))
            textSize = 16f
            setPadding(dp(24), dp(20), dp(24), dp(8))
        }
        // the same rows as setItems, but a tap on a link in a description opens it instead of choosing the model
        val rows = object : ArrayAdapter<CharSequence>(this, com.google.android.material.R.layout.mtrl_alert_select_dialog_item, android.R.id.text1, items) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View =
                super.getView(position, convertView, parent).also { v -> (v as? TextView)?.let { Links.openOnTap(it) } }
        }
        MaterialAlertDialogBuilder(this)
            .setCustomTitle(title)
            .setAdapter(rows) { _, which ->
                modelName = ModelRegistry.models[which].name
                showModel()
                refresh()
            }
            .show()
    }

    private fun showModel() {
        val m = modelName?.let { ModelRegistry.find(it) }
        if (m != null) {
            modelText.text = twoLine(m.title, m.description)
            modelText.setTextColor(getColor(R.color.ink))
        } else {
            modelText.setText(R.string.pick_model)
            modelText.setTextColor(getColor(R.color.placeholder))
        }
    }

    private fun twoLine(title: String, sub: String): CharSequence {
        val sb = SpannableStringBuilder(title).append('\n')
        val start = sb.length
        sb.append(sub)
        sb.setSpan(ForegroundColorSpan(getColor(R.color.muted)), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        sb.setSpan(RelativeSizeSpan(0.85f), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        return Links.linkify(sb)
    }

    // ------------------------------------------------------------------ More options

    private val watcher = object : TextWatcher {
        override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        override fun afterTextChanged(s: Editable?) = refresh()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun section(parent: LinearLayout, title: String) {
        parent.addView(TextView(this, null, 0, R.style.Section).apply {
            text = title
            setPadding(0, dp(14), 0, 0)
        })
    }

    /** One-sentence explanation under an option: what a higher / lower value, or ticking the box, does. */
    private fun help(parent: LinearLayout, text: String) {
        parent.addView(TextView(this).apply {
            this.text = text
            textSize = 12f
            setTextColor(getColor(R.color.muted))
            setPadding(dp(4), dp(2), dp(4), dp(2))
        })
    }

    private fun field(parent: LinearLayout, label: String, value: String, number: Boolean = true, help: String? = null): TextInputEditText {
        val til = TextInputLayout(this, null, com.google.android.material.R.attr.textInputOutlinedStyle).apply {
            hint = label
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { topMargin = dp(6) }
        }
        val et = TextInputEditText(til.context).apply {
            setText(value)
            inputType = if (number) InputType.TYPE_CLASS_NUMBER else InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            isSingleLine = true
            addTextChangedListener(watcher)
        }
        til.addView(et)
        parent.addView(til)
        help?.let { help(parent, it) }
        return et
    }

    private fun check(parent: LinearLayout, label: String, help: String? = null): MaterialCheckBox =
        MaterialCheckBox(this).apply {
            text = label
            setOnCheckedChangeListener { _, _ -> showModel(); refresh() }
            parent.addView(this)
            help?.let { help(parent, it) }
        }

    private fun dropdown(parent: LinearLayout, label: String, items: Array<String>, value: String, help: String? = null): MaterialAutoCompleteTextView {
        val til = TextInputLayout(this, null, com.google.android.material.R.attr.textInputOutlinedExposedDropdownMenuStyle).apply {
            hint = label
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { topMargin = dp(6) }
        }
        val dd = MaterialAutoCompleteTextView(til.context).apply {
            inputType = InputType.TYPE_NULL
            setSimpleItems(items)
            setText(value, false)
            addTextChangedListener(watcher)
        }
        til.addView(dd)
        parent.addView(til)
        help?.let { help(parent, it) }
        return dd
    }

    private fun buildMoreOptions(p: LinearLayout) {
        section(p, "CLEAN MOSAIC")
        oMaskThreshold = field(p, "Detection threshold 0~255 (--mask_threshold)", "128",
            help = "Higher only cleans spots the detector is sure about (fewer false spots, may miss faint mosaics); lower also cleans weaker detections.")
        oMedfilt = field(p, "Video smoothing window, odd (--medfilt_num)", "11",
            help = "Higher keeps the cleaned area steadier over time but slower to follow movement; lower follows fast motion but may jitter.")
        oMinMosaic = field(p, "Smallest mosaic to clean, px; 0 = model default (--min_mosaic_size)", "0",
            help = "Higher skips small mosaics so only big ones get cleaned; lower also cleans tiny ones; 0 uses the model's default (32 px).")
        oExMult = field(p, "Mosaic area expansion, auto or number (--ex_mult)", "auto", number = false,
            help = "Higher repaints a wider margin around each mosaic (safer edges); lower stays tight around it; auto uses 1.5, what the restorer was trained on.")
        oNoFeather = check(p, "No edge feather, faster (--no_feather)",
            help = "Ticked: pastes the cleaned patch with a hard edge instead of blending it in (slightly faster, the edge may show).")
        oAllArea = check(p, "Find all mosaic areas (--all_mosaic_area)",
            help = "Ticked: cleans every mosaic found in a frame instead of only the largest one.")

        section(p, "SHARPEN CLEANED AREA (REAL-ESRGAN)")
        oEnhancePatch = check(p, "Sharpen the cleaned area only (--enhance_patch)",
            help = "Ticked: Real-ESRGAN sharpens the repaired patch before it is blended back (crisper, a few seconds more per video frame).")
        oSrModel = dropdown(p, "Real-ESRGAN model (--sr_model)", Options.SR_MODELS.reversed().toTypedArray(), Options.SR_ANIME,
            help = "realesr-animevideov3 gives a smooth, clean result (best for animation); realesr-general-x4v3 keeps more real-photo texture.")
        oDenoise = dropdown(p, "Denoise strength, realesr-general only (--denoise_strength)", DENOISE_ITEMS, DENOISE_ITEMS[1],
            help = "realesr-general only: 0 keeps grain and fine detail, 1 removes noise for a smoother look, 0.5 sits in between.")
        oParallelSharpen = check(p, "Sharpen in parallel on the spare cores, videos (--parallel_sharpen)",
            help = "Ticked: sharpening runs on the spare cores while the next frames are cleaned (same result, usually much less waiting).").apply { isChecked = true }

        section(p, "VIDEO")
        oStart = field(p, "Start time HH:MM:SS (-ss)", "00:00:00", number = false,
            help = "Where processing begins: a later time skips more of the video's start.")
        oDuration = field(p, "Duration HH:MM:SS, 0 = all (-t)", "00:00:00", number = false,
            help = "How much to process from the start time: shorter is a quicker test, 0 runs to the end.")
        oMaxVideo = field(p, "Max video size (short side), 0 = original", "0",
            help = "Lower (e.g. 720) processes faster at a reduced resolution; 0 keeps the video's own size.")

        section(p, "PERFORMANCE")
        oThreads = field(p, "AI threads, 0 = auto: one per performance core (--threads)", "0",
            help = "More threads is faster up to the number of fast cores but runs hotter; fewer is cooler and slower; 0 picks for this phone.")
        oAffinity = check(p, "Bind AI to performance cores (off: --no_affinity)",
            help = "Ticked: keeps the AI on the phone's fast cores, which is usually faster and steadier.").apply { isChecked = true }
        oParallelColor = check(p, "Split colour conversion over the spare cores (--parallel_color)",
            help = "Ticked: spreads each frame's colour conversion over the spare cores; helps from 720p up and with HDR videos.")
        p.addView(TextView(this).apply {
            val topo = CpuTopology.current
            text = getString(R.string.detected_cpu, topo.describe())
            textSize = 12f
            setTextColor(getColor(R.color.muted))
            setPadding(0, dp(2), 0, 0)
        })

        section(p, "MORE")
        oExtra = field(p, getString(R.string.extra_options_hint), "", number = false,
            help = "Extra --options typed as text, applied on top of everything above; leave it empty if unsure.")
        cmd = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.code_ink))
            setBackgroundColor(getColor(R.color.code_bg))
            setPadding(dp(6), dp(6), dp(6), dp(6))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { topMargin = dp(10) }
        }
        p.addView(cmd)
    }

    private fun int(et: TextInputEditText, name: String): Int =
        et.text?.toString()?.trim()?.ifEmpty { "0" }?.toIntOrNull() ?: throw OptionError("$name must be a whole number")

    /** Builds Options from the form, then applies the free-text options on top (like the Windows text box). */
    private fun buildOptions(): Options {
        val o = Options(
            mode = mode.text.toString().lowercase(),
            modelName = modelName ?: "",
            fps = int(fps, "FPS"),
            startTime = Options.parseTime(oStart.text.toString().ifBlank { "0" }),
            lastTime = Options.parseTime(oDuration.text.toString().ifBlank { "0" }),
            maskThreshold = oMaskThreshold.text.toString().trim().ifEmpty { "128" }.toIntOrNull()
                ?: throw OptionError("Detection threshold must be a whole number"),
            enhancePatch = oEnhancePatch.isChecked,
            srModel = oSrModel.text.toString(),
            denoiseStrength = DENOISE_VALUES[DENOISE_ITEMS.indexOf(oDenoise.text.toString()).coerceAtLeast(0)],
            noFeather = oNoFeather.isChecked,
            allMosaicArea = oAllArea.isChecked,
            medfiltNum = int(oMedfilt, "medfilt_num"),
            minMosaicSize = int(oMinMosaic, "min_mosaic_size"),
            exMult = oExMult.text.toString().trim().ifEmpty { "auto" },
            maxVideoSize = int(oMaxVideo, "Max video size"),
            threads = int(oThreads, "AI threads"),
            affinity = oAffinity.isChecked,
            parallelSharpen = oParallelSharpen.isChecked,
            parallelColor = oParallelColor.isChecked,
        )
        Options.applyArgs(o, oExtra.text.toString())
        return o
    }

    private fun refresh() {
        if (!::cmd.isInitialized) return
        cmd.text = try {
            buildOptions().toCommandLine(mediaName.ifEmpty { "<file>" })
        } catch (e: OptionError) {
            "⚠ ${e.message}"
        }
        hint.text = when {
            Jobs.isBusy -> "A job is running. Tap Run! to see it."
            mediaUri == null -> noMediaHint()
            modelName == null -> "Choose a model"
            else -> ""
        }
    }

    // ------------------------------------------------------------------ Run

    private fun onRun() {
        if (Jobs.isBusy) {
            startActivity(Intent(this, RunActivity::class.java))
            return
        }
        if (mediaUri == null) { hint.text = noMediaHint(); return }
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED &&
            !prefs.getBoolean("asked_notifications", false)
        ) {
            prefs.edit().putBoolean("asked_notifications", true).apply()
            askNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        checkBatteryThenRun()
    }

    /**
     * Once: ask Android to leave MosaicGone out of battery optimisation. With it on, Android (and Samsung's battery
     * management above all) may end a long job's process when the app is not on screen, and the job is lost.
     */
    // lint (BatteryLife): Google Play's policy limits this request. MosaicGone is distributed as an APK, not on Play, and
    // a long video job is exactly what the exemption is for; revisit before any Play Store release
    @SuppressLint("BatteryLife")
    private fun checkBatteryThenRun() {
        val pm = getSystemService(android.os.PowerManager::class.java)
        if (pm == null || pm.isIgnoringBatteryOptimizations(packageName) || prefs.getBoolean("asked_battery", false)) {
            startRun()
            return
        }
        prefs.edit().putBoolean("asked_battery", true).apply()
        MaterialAlertDialogBuilder(this)
            .setTitle("Keep jobs running in the background")
            .setMessage(
                "Cleaning a video can take a while. So that Android does not stop MosaicGone when you leave the app, " +
                    "allow it to run in the background on the next screen (you can change this later in the app's battery settings).",
            )
            .setPositiveButton("Continue") { _, _ ->
                val ok = runCatching {
                    askBattery.launch(
                        Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:$packageName")),
                    )
                }.isSuccess
                if (!ok) startRun()
            }
            .setNegativeButton("Not now") { _, _ -> startRun() }
            .setCancelable(false)
            .show()
    }

    private fun startRun() {
        val uri = mediaUri ?: return
        val opt = try {
            buildOptions().resolve()
        } catch (e: OptionError) {
            MaterialAlertDialogBuilder(this).setTitle("Please check the options").setMessage(e.message).setPositiveButton("OK", null).show()
            return
        }
        savePrefs()
        ProcessingService.start(this, JobRequest(uri, mediaIsVideo, mediaName, opt))
        // the model is chosen again for every job: back on this screen it asks for one
        modelName = null
        showModel()
        startActivity(Intent(this, RunActivity::class.java))
    }

    private fun savePrefs() {
        prefs.edit().apply {
            putString("mode", mode.text.toString())
            putString("fps", fps.text.toString())
            putString("sr_model", oSrModel.text.toString())
            putString("denoise", oDenoise.text.toString())
            for ((k, et) in textFields()) putString(k, et.text.toString())
            for ((k, cb) in checkBoxes()) putBoolean(k, cb.isChecked)
            putInt("prefs_version", 6)
        }.apply()
    }

    private fun restorePrefs() {
        prefs.getString("mode", null)?.takeIf { it in MODES }?.let { mode.setText(it, false) }
        prefs.getString("fps", null)?.let { fps.setText(it) }
        // the model is never remembered: every job starts with "Choose a model"
        prefs.edit().remove("model").apply()
        prefs.getString("sr_model", null)?.takeIf { it in Options.SR_MODELS }?.let { oSrModel.setText(it, false) }
        prefs.getString("denoise", null)?.takeIf { it in DENOISE_ITEMS }?.let { oDenoise.setText(it, false) }
        for ((k, et) in textFields()) prefs.getString(k, null)?.let { et.setText(it) }
        // an earlier v0.1 build saved "auto" here
        if (oMaskThreshold.text.toString().trim().toIntOrNull() == null) oMaskThreshold.setText(R.string.default_mask_threshold)
        oEnhancePatch.isChecked = prefs.getBoolean("enhance_patch", false)
        oNoFeather.isChecked = prefs.getBoolean("no_feather", false)
        oAllArea.isChecked = prefs.getBoolean("all_mosaic_area", false)
        oAffinity.isChecked = prefs.getBoolean("affinity", true)
        oParallelSharpen.isChecked = prefs.getBoolean("parallel_sharpen", true)
        oParallelColor.isChecked = prefs.getBoolean("parallel_color", false)
        showModel()
    }

    private fun textFields() = listOf(
        "mask_threshold" to oMaskThreshold,
        "medfilt_num" to oMedfilt, "min_mosaic_size" to oMinMosaic, "ex_mult" to oExMult,
        "start" to oStart, "duration" to oDuration, "max_video" to oMaxVideo,
        "threads" to oThreads, "extra" to oExtra,
    )

    private fun checkBoxes() = listOf(
        "enhance_patch" to oEnhancePatch, "no_feather" to oNoFeather, "all_mosaic_area" to oAllArea,
        "affinity" to oAffinity, "parallel_sharpen" to oParallelSharpen, "parallel_color" to oParallelColor,
    )

    // ------------------------------------------------------------------ Auto-Select Options

    /** Reads this phone, the chosen file, model and FPS, then fills More Options (see [AutoTune]) and says why. */
    private fun autoSelect() {
        val uri = mediaUri
        if (uri == null) { hint.text = noMediaHint(); return }
        val current = try {
            // no model picked yet: tune for the restorer (the one the estimate mentions)
            buildOptions().let { if (it.modelName.isEmpty()) it.copy(modelName = Options.RESTORER) else it }
        } catch (e: OptionError) {
            MaterialAlertDialogBuilder(this).setTitle("Please check the options").setMessage(e.message).setPositiveButton("OK", null).show()
            return
        }
        hint.setText(R.string.reading_phone_and_file)
        lifecycleScope.launch {
            val r = withContext(Dispatchers.IO) { runCatching { AutoTune.choose(deviceSpec(), mediaSpec(uri), current) } }
            hint.text = ""
            r.onFailure {
                MaterialAlertDialogBuilder(this@MainActivity).setTitle("Auto-Select Options").setMessage("Could not read the file: ${it.message}").setPositiveButton("OK", null).show()
            }
            r.onSuccess { res ->
                applyOptions(res.options)
                findViewById<MaterialCheckBox>(R.id.moreCheck).isChecked = true
                val topo = CpuTopology.current
                val msg = buildString {
                    append("For this phone (${topo.describe()}) and ${mediaName.ifEmpty { "this file" }}:\n\n")
                    res.reasons.forEach { append("• ").append(it).append('\n') }
                    res.estimateSec?.let { append("\nEstimated time: ${AutoTune.fmtEstimate(it)} (rough, assumes a mosaic in every frame).") }
                    if (modelName == null) append("\n\nNo model chosen yet: estimates assume ${ModelRegistry.RESTORER_TITLE}.")
                    append("\n\nYou can still change any option in More Options.")
                }
                MaterialAlertDialogBuilder(this@MainActivity).setTitle("Options selected").setMessage(msg).setPositiveButton("OK", null).show()
                refresh()
            }
        }
    }

    private fun deviceSpec(): AutoTune.Device {
        val topo = CpuTopology.current
        val am = getSystemService(android.app.ActivityManager::class.java)
        val mem = android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        fun ghz(cs: List<Int>) = cs.mapNotNull { topo.maxKHz[it] }.maxOrNull()?.div(1e6) ?: 2.0
        return AutoTune.Device(
            perfCores = topo.perf.size, effCores = topo.eff.size, perfGHz = ghz(topo.perf), effGHz = ghz(topo.eff),
            heterogeneous = topo.heterogeneous, ramGB = mem.totalMem / 1e9, heapMB = Runtime.getRuntime().maxMemory() shr 20,
        )
    }

    private fun mediaSpec(uri: Uri): AutoTune.Media = when {
        Media.isVideo(this, uri) -> VideoProbe.probe(this, uri).let {
            AutoTune.Media(AutoTune.Kind.VIDEO, it.displayWidth, it.displayHeight, it.fps, it.durationUs / 1e6, it.color.spec.isHdr, it.tenBit)
        }
        Media.isGif(this, uri) -> Gif.decode(Media.readBytes(this, uri)).let { a ->
            val sec = a.frames.sumOf { it.delayCs } / 100.0
            AutoTune.Media(AutoTune.Kind.GIF, a.width, a.height, if (sec > 0) a.frames.size / sec else 10.0, sec)
        }
        else -> {
            val o = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { android.graphics.BitmapFactory.decodeStream(it, null, o) }
            AutoTune.Media(AutoTune.Kind.IMAGE, o.outWidth.coerceAtLeast(1), o.outHeight.coerceAtLeast(1))
        }
    }

    /** Writes the More Options part of [o] into the form (model, FPS, start/duration and the text box are untouched). */
    // lint (SetTextI18n) asks for locale-formatted numbers here; these boxes are read back with toIntOrNull(), which only
    // understands plain ASCII digits, so locale digits (e.g. Arabic) would break the options: plain toString() is right
    @SuppressLint("SetTextI18n")
    private fun applyOptions(o: Options) {
        oMaskThreshold.setText(o.maskThreshold.toString())
        oMedfilt.setText(o.medfiltNum.toString())
        oMinMosaic.setText(o.minMosaicSize.toString())
        oExMult.setText(o.exMult)
        oNoFeather.isChecked = o.noFeather
        oAllArea.isChecked = o.allMosaicArea
        oEnhancePatch.isChecked = o.enhancePatch
        oSrModel.setText(o.srModel, false)
        oDenoise.setText(DENOISE_ITEMS[DENOISE_VALUES.indexOfFirst { it == o.denoiseLevel }.coerceAtLeast(0)], false)
        oMaxVideo.setText(o.maxVideoSize.toString())
        oThreads.setText(o.threads.toString())
        oAffinity.isChecked = o.affinity
        oParallelSharpen.isChecked = o.parallelSharpen
        oParallelColor.isChecked = o.parallelColor
    }

    private companion object {
        val MODES = listOf("Auto", "Clean")
        val DENOISE_ITEMS = arrayOf("0 · keep noise and detail", "0.5 · balanced", "1 · strong denoise")
        val DENOISE_VALUES = doubleArrayOf(0.0, 0.5, 1.0)
    }

    // ------------------------------------------------------------------ Help / About

    private fun showHelp() {
        MaterialAlertDialogBuilder(this)
            .setTitle("Help")
            .setMessage(
                """
                1. Choose an image or video: from the Gallery, or from Files (any folder, SD card, USB drive or cloud app).
                2. Choose the model.
                3. Optional: tap Auto-Select Options. It reads this phone, the file and the FPS, and fills More Options with the best balance for this phone. It explains each choice; you can still change any of them.
                4. Tap Run! and wait.
                5. Find the result in your Gallery: Pictures/MosaicGone or Movies/MosaicGone.

                Models
                • ${ModelRegistry.DETECTOR_TITLE}: finds the mosaics in every frame.
                • ${ModelRegistry.RESTORER_TITLE}: restores what the mosaic hides, for photos, GIFs and videos, real footage and animation. It cleans each stretch of video 16 frames at a time, looking backwards and forwards, so the cleaned area stays steady.
                • ${ModelRegistry.LADA_TITLE}: Lada's full-size restorer, used the same way. Cleaner on animation faces, about the same on real footage, about 2x slower.

                Sharpen cleaned area (Real-ESRGAN, off by default)
                • More Options → "Sharpen the cleaned area only". Real-ESRGAN runs on the repaired patch before it is blended back, so the cleaned area is less blurry. The rest of the picture or video is never changed.
                • Costs a few extra seconds per video frame (see "enhance" in the log's timing table).
                • Model: realesr-animevideov3 (default, smooth result) or realesr-general-x4v3 (with a denoise strength: 0 keeps grain, 1 removes it).

                Output quality
                • Videos are encoded at the source's bitrate (HEVC if the source is HEVC), with the source's colour standard and range.
                • Photos keep their format and bits per pixel (PNG stays lossless; HEIC is saved as JPEG at 2× its bits per pixel) and their colour profile (e.g. Display P3).
                • Animated GIFs are cleaned frame by frame and saved as GIF with the source's palettes, size and timing.
                • No size or length limit: pictures and videos are processed at their own size. Only if the phone cannot spare the memory is the size reduced, and the log says so.
                • 10-bit and HDR (HLG / PQ) videos are read at full precision and converted to standard (SDR) colours.
                • Readable videos: MP4, MOV, 3GP, MKV, WebM, MPEG-TS (H.264, HEVC, VP8/VP9, AV1 where the phone has a decoder). AVI, WMV and FLV need converting first.

                Tips for phones
                • Video is slow (a few seconds per frame). Set FPS to 10–15 and use Start time / Duration to process a short part first.
                • "Max video size" (default 0 = the source's size): set e.g. 720 to process big videos faster at a lower resolution.
                • Processing keeps running with the screen off; follow it in the notification.
                • The More Options text box accepts the same --options as the fields above (e.g. --fps 10).

                Measuring speed
                • Every job writes a log with a per-step timing breakdown (read/decode, pre-processing, AI, post-processing, encoding), the CPU layout, thermal state and clock speeds. Tap its name on the result screen; it is saved in Download/MosaicGone.
                • AI threads (0 = one per performance core) and "Bind AI to performance cores" are in More Options → Performance.
                • --no_parallel runs image handling on the AI thread instead of in parallel, for A/B comparisons.
                • "Sharpen in parallel" (--parallel_sharpen, on by default): with sharpening on, Real-ESRGAN runs on the spare (efficiency) cores while the restorer works on the next frames, and the AI thread helps when they fall behind. Same result, much less extra time.
                • "Split colour conversion" (--parallel_color, off by default): spreads the per-frame colour conversion over the spare cores; helps from 720p up and with HDR.
                • The log lists which stretches of a video were cleaned and which were left untouched, and why (no mosaic found, too small, or dropped by the smoothing), with timestamps, and the sizes of the mosaics rejected as too small.
                • "Smallest mosaic to clean" (--min_mosaic_size, 0 = the restorer's default of 32 px): raise it to leave small mosaics alone.
                • If another app takes the phone's video decoder or encoder while a job runs (e.g. watching videos), the job continues with another one. If a job still fails, the part finished so far is saved as …_clean_partial.mp4.
                """.trimIndent(),
            )
            .setPositiveButton("OK", null)
            .show()
    }

    private fun showAbout() {
        val version = packageManager.getPackageInfo(packageName, 0).versionName
        showWithLinks(
            "About",
            "MosaicGone v$version\n\n" +
                "This app finds and removes mosaics in pictures and videos, running fully offline on your device " +
                "(ONNX Runtime, CPU). MosaicGone's own trained models are YOLOX-Nano for mosaic detection and BasicVSR++ " +
                "scaled down to a size that allows effective performance on mobile devices. This app also includes the " +
                "original Lada restoration model for those who wish to use that model instead, but it is not optimized to " +
                "run on mobile; we cannot guarantee its performance. We've also included sharpening effects using " +
                "Real-ESRGAN as an additional feature.\n\n" +
                "See Credits for a full list of every project this app was built on. Thanks, and enjoy!",
        )
    }

    /** Where to find the author's other projects and how to support the app; same bubble as About. */
    private fun showSupport() {
        showWithLinks(
            "Support",
            "• View the Source Code of this app at GitHub: https://github.com/damzenquis/MosaicGone\n\n" +
                "• View my projects at GitHub: https://github.com/damzenquis\n\n" +
                "• Buy me a coffee and drop an idea while you're at it: https://ko-fi.com/damzenquis",
        )
    }

    /** The projects and models MosaicGone took from: one line each, what it did for us. Same bubble as About. */
    private fun showCredits() {
        showWithLinks(
            "Credits",
            """
            • DeepMosaics by HypoX64 (https://github.com/HypoX64/DeepMosaics) (GPL-3.0) = Used to develop the pipeline for ensuring the scope of this project worked on Android.

            • Lada by ladaapp (https://github.com/ladaapp/lada) (AGPL-3.0) = Its original restoration model is included as a second restorer, and its models and dataset tools guided our training.

            • Real-ESRGAN by Xintao Wang et al. (https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) = Included as a feature to allow users to sharpen the media selected.

            • YOLOX by Megvii (https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) = The design and starting weights of our YOLOX-Nano mosaic detector.

            • BasicVSR++ by Kelvin C.K. Chan et al. (https://github.com/ckkelvinchan/BasicVSR_PlusPlus) (Apache-2.0) = The design of our restorer, scaled down to run on mobile devices.

            • MMagic by OpenMMLab (https://github.com/open-mmlab/mmagic) (Apache-2.0) = Its SPyNet motion-estimation weights are part of our restorer.

            • APISR by Boyang Wang et al. (https://github.com/Kiteretsu77/APISR) (GPL-3.0) = Its ideas shaped how we trained our restorer on animation.

            • Danbooru pretrained ResNet50 by Matthew Baas (https://github.com/RF5/danbooru-pretrained) (MIT, adapted) = Used to judge how anime-like our restorer's results looked during training.
            """.trimIndent(),
        )
    }

    /** Source, license and contact (the AGPL's "Appropriate Legal Notices"), with the full license and notices bundled in assets/legal. */
    private fun showLicense() {
        val version = packageManager.getPackageInfo(packageName, 0).versionName
        MaterialAlertDialogBuilder(this)
            .setTitle("Source, License & Contact")
            .setMessage(
                Links.linkify(
                    "MosaicGone v$version\nCopyright (C) 2026 damzenquis\n\n" +
                        "MosaicGone is free software: you can redistribute it and/or modify it under the terms of the GNU " +
                        "Affero General Public License, version 3 (AGPL-3.0), as published by the Free Software Foundation.\n\n" +
                        "MosaicGone is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even " +
                        "the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero " +
                        "General Public License (Full license, below) for more details.\n\n" +
                        "It includes models and code from other projects, each under its own license: Lada (AGPL-3.0), " +
                        "DeepMosaics (GPL-3.0), Real-ESRGAN (BSD-3-Clause), YOLOX, MMagic and the Android and Kotlin " +
                        "libraries (Apache-2.0), ONNX Runtime (MIT). See Credits, and Notices below.\n\n" +
                        "Source code, including the models and how they were made: https://github.com/damzenquis/MosaicGone\n" +
                        "Questions and bug reports: https://github.com/damzenquis",
                ),
            )
            .setPositiveButton("OK", null)
            .setNeutralButton("Full license") { _, _ -> showLegalFile("GNU AGPL-3.0", "legal/LICENSE.txt") }
            .setNegativeButton("Notices") { _, _ -> chooseNotice() }
            .show()
            .also { Links.enable(it) }
    }

    /** The bundled notices: THIRD_PARTY_NOTICES.md first, then every license text in third_party_licenses/. */
    private fun chooseNotice() {
        fun files(dir: String): List<String> = assets.list(dir).orEmpty().sorted().flatMap { name ->
            val path = "$dir/$name"
            if (assets.list(path).isNullOrEmpty()) listOf(path) else files(path)
        }
        val paths = listOf("legal/THIRD_PARTY_NOTICES.md") + files("legal/third_party_licenses")
        // "legal/third_party_licenses/real-esrgan/LICENSE" -> "real-esrgan · LICENSE"
        val labels = paths.map { it.removePrefix("legal/").removePrefix("third_party_licenses/").replace("/", " · ") }
        MaterialAlertDialogBuilder(this)
            .setTitle("Third-party notices")
            .setItems(labels.toTypedArray()) { _, i -> showLegalFile(labels[i], paths[i]) }
            .setPositiveButton("Close", null)
            .show()
    }

    private fun showLegalFile(title: String, path: String) {
        lifecycleScope.launch {
            val text = withContext(Dispatchers.IO) { runCatching { assets.open(path).bufferedReader().use { it.readText() } } }
            text.onSuccess { showWithLinks(title, it) }
                .onFailure { showWithLinks(title, "Could not open $path: ${it.message}") }
        }
    }

    private fun showWithLinks(title: String, text: String) {
        MaterialAlertDialogBuilder(this)
            .setTitle(title)
            .setMessage(Links.linkify(text))
            .setPositiveButton("OK", null)
            .show()
            .also { Links.enable(it) }
    }
}
