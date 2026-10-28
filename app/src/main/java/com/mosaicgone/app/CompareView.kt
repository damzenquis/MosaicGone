package com.mosaicgone.app

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.min

/** Before/after viewer: drag left/right to move the divider. */
class CompareView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {

    var before: Bitmap? = null
        set(v) { field = v; invalidate() }
    var after: Bitmap? = null
        set(v) { field = v; invalidate() }

    private var split = 0.5f
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val line = Paint().apply { color = Color.WHITE; strokeWidth = 3 * resources.displayMetrics.density }
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 12 * resources.displayMetrics.scaledDensity }
    private val labelBg = Paint().apply { color = 0x88000000.toInt() }

    private fun dest(b: Bitmap): RectF {
        val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
        val w = b.width * s
        val h = b.height * s
        return RectF((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
    }

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(context.getColor(R.color.preview_bg))
        val a = after ?: return
        val r = dest(a)
        val x = r.left + r.width() * split
        before?.let {
            canvas.save()
            canvas.clipRect(r.left, r.top, x, r.bottom)
            canvas.drawBitmap(it, null, r, paint)
            canvas.restore()
        }
        canvas.save()
        canvas.clipRect(if (before != null) x else r.left, r.top, r.right, r.bottom)
        canvas.drawBitmap(a, null, r, paint)
        canvas.restore()
        if (before != null) {
            canvas.drawLine(x, r.top, x, r.bottom, line)
            tag(canvas, "Before", r.left + 6, r.top + 6, left = true)
            tag(canvas, "After", r.right - 6, r.top + 6, left = false)
        }
    }

    private fun tag(canvas: Canvas, text: String, x: Float, y: Float, left: Boolean) {
        val w = label.measureText(text) + 12
        val h = label.textSize + 8
        val l = if (left) x else x - w
        canvas.drawRect(l, y, l + w, y + h, labelBg)
        canvas.drawText(text, l + 6, y + h - 6, label)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        val a = after ?: return false
        val r = dest(a)
        if (e.action == MotionEvent.ACTION_DOWN) parent.requestDisallowInterceptTouchEvent(true)
        split = ((e.x - r.left) / r.width()).coerceIn(0f, 1f)
        invalidate()
        return true
    }
}
