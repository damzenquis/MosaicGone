package com.mosaicgone.app

import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.text.Spannable
import android.text.SpannableString
import android.text.Spanned
import android.text.method.LinkMovementMethod
import android.text.style.URLSpan
import android.view.MotionEvent
import android.widget.TextView
import androidx.appcompat.app.AlertDialog

/** The web links written in the app's texts (About, Credits, model descriptions), made tappable. */
object Links {
    private val URL = Regex("""https?://[^\s()<>]+""")

    /** where each link is in [text]; a full stop or comma right after a link is not part of it */
    fun find(text: CharSequence): List<IntRange> = URL.findAll(text).map { m ->
        var end = m.range.last
        while (end > m.range.first && text[end] in ".,;:!?'\"") end--
        m.range.first..end
    }.toList()

    fun linkify(text: CharSequence): Spannable {
        val s = text as? Spannable ?: SpannableString(text)
        for (r in find(s)) s.setSpan(URLSpan(s.substring(r)), r.first, r.last + 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        return s
    }

    /** for a dialog shown with [linkify]-ed text: links open in the browser */
    fun enable(dialog: AlertDialog) {
        dialog.findViewById<TextView>(android.R.id.message)?.movementMethod = LinkMovementMethod.getInstance()
    }

    /**
     * For text inside something tappable (the Step 2 card, a row in the model list): a tap on a link opens it, a tap
     * anywhere else still reaches the card or row.
     */
    @SuppressLint("ClickableViewAccessibility")
    fun openOnTap(tv: TextView) {
        tv.setOnTouchListener { v, e ->
            val t = v as TextView
            val text = t.text as? Spanned ?: return@setOnTouchListener false
            val layout = t.layout ?: return@setOnTouchListener false
            val line = layout.getLineForVertical(e.y.toInt() - t.totalPaddingTop + t.scrollY)
            val x = e.x - t.totalPaddingLeft + t.scrollX
            if (x < layout.getLineLeft(line) || x > layout.getLineRight(line)) return@setOnTouchListener false
            val off = layout.getOffsetForHorizontal(line, x)
            val link = text.getSpans(off, off, URLSpan::class.java).firstOrNull() ?: return@setOnTouchListener false
            if (e.action == MotionEvent.ACTION_UP) open(t, link.url)
            true
        }
    }

    private fun open(v: TextView, url: String) {
        try {
            v.context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
        } catch (_: ActivityNotFoundException) {
        }
    }
}
