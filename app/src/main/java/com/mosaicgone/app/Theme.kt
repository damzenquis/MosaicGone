package com.mosaicgone.app

import android.app.Application
import android.content.Context
import androidx.appcompat.app.AppCompatDelegate

/**
 * Light or dark, picked with the chip on the main screen's top bar. The only setting kept when the app is closed;
 * applied when the process starts, before any screen, so a job's screen opened from the notification matches too.
 */
object Theme {
    private const val PREFS = "theme"
    private const val KEY_DARK = "dark"

    fun isDark(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_DARK, false)

    fun apply(dark: Boolean) = AppCompatDelegate.setDefaultNightMode(
        if (dark) AppCompatDelegate.MODE_NIGHT_YES else AppCompatDelegate.MODE_NIGHT_NO,
    )

    /** saves the choice and switches every open screen to it (Android recreates them) */
    fun set(ctx: Context, dark: Boolean) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_DARK, dark).apply()
        apply(dark)
    }
}

class MosaicGoneApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Theme.apply(Theme.isDark(this))
    }
}
