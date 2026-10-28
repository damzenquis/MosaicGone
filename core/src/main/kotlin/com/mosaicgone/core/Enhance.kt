// Modified for MosaicGone, 2026: translated to Kotlin from realesrgan/utils.py (RealESRGANer.enhance) of Real-ESRGAN,
// https://github.com/xinntao/Real-ESRGAN, under the following license:
//
// BSD 3-Clause License
//
// Copyright (c) 2021, Xintao Wang
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
package com.mosaicgone.core

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Port of realesrgan.utils.RealESRGANer.enhance for the bundled x4 SRVGGNetCompact models
 * (fp32, pre_pad 0 like inference_realesrgan.py, 8-bit RGB; no alpha / 16-bit / face enhance).
 *
 * In this app it only sharpens the repaired patch of a clean (Pipeline.sharpen), never a whole picture or video.
 *
 * Split in three parts (kept from the port; tiling is unused for the small patches):
 *   [prepare]  cut the image into tiles (with tile_pad context) and build RGB 0..1 tensors
 *   [infer]    run the network per tile, crop away the padding, clamp/round to 8-bit
 *   [assemble] paste the tiles into the x4 image and resize to outscale (Lanczos4, as in Python)
 */
class Enhancer(private val net: Net, private val tile: Int = 0, private val tilePad: Int = 10) {

    companion object {
        const val SCALE = 4
    }

    /** Input tile area [x0,x1)x[y0,y1) and the padded area [px0,px1)x[py0,py1) that goes through the net. */
    class Tile(val x0: Int, val y0: Int, val x1: Int, val y1: Int, val px0: Int, val py0: Int, val px1: Int, val py1: Int) {
        var input: Net.Tensor? = null
    }

    class Job(val w: Int, val h: Int, val tiles: List<Tile>)

    fun prepare(img: Img): Job {
        val w = img.w
        val h = img.h
        val tiles = ArrayList<Tile>()
        if (tile <= 0) {
            tiles.add(Tile(0, 0, w, h, 0, 0, w, h))
        } else {
            val tx = ceil(w / tile.toDouble()).toInt()
            val ty = ceil(h / tile.toDouble()).toInt()
            for (y in 0 until ty) for (x in 0 until tx) {
                val sx = x * tile
                val sy = y * tile
                val ex = min(sx + tile, w)
                val ey = min(sy + tile, h)
                tiles.add(Tile(sx, sy, ex, ey, max(sx - tilePad, 0), max(sy - tilePad, 0), min(ex + tilePad, w), min(ey + tilePad, h)))
            }
        }
        for (t in tiles) t.input = toRgb01(img, t.px0, t.py0, t.px1 - t.px0, t.py1 - t.py0)
        return Job(w, h, tiles)
    }

    private fun toRgb01(img: Img, x0: Int, y0: Int, tw: Int, th: Int): Net.Tensor {
        val n = tw * th
        val out = FloatArray(3 * n)
        for (y in 0 until th) {
            var p = ((y0 + y) * img.w + x0) * 3
            for (x in 0 until tw) {
                val q = y * tw + x
                out[q] = (img.data[p].toInt() and 0xFF) / 255f
                out[n + q] = (img.data[p + 1].toInt() and 0xFF) / 255f
                out[2 * n + q] = (img.data[p + 2].toInt() and 0xFF) / 255f
                p += 3
            }
        }
        return Net.Tensor(out, longArrayOf(1, 3, th.toLong(), tw.toLong()))
    }

    /** Runs every tile; returns the x4 output of each tile without its padding, as 8-bit RGB. */
    fun infer(job: Job): List<Img> = job.tiles.map { t ->
        val y = net.run("input" to t.input!!)
        t.input = null
        val ow = y.shape[3].toInt()
        val oh = y.shape[2].toInt()
        val cw = (t.x1 - t.x0) * SCALE
        val ch = (t.y1 - t.y0) * SCALE
        val ox = (t.x0 - t.px0) * SCALE
        val oy = (t.y0 - t.py0) * SCALE
        val n = ow * oh
        val part = Img(cw, ch, 3)
        for (yy in 0 until ch) for (xx in 0 until cw) {
            val s = (oy + yy) * ow + ox + xx
            val q = (yy * cw + xx) * 3
            for (c in 0 until 3) {
                // output.clamp_(0, 1) then (output * 255.0).round().astype(uint8)
                part.data[q + c] = Math.rint((y.data[c * n + s].coerceIn(0f, 1f) * 255f).toDouble()).toInt().toByte()
            }
        }
        part
    }

    fun assemble(job: Job, parts: List<Img>, outscale: Double): Img {
        val out = Img(job.w * SCALE, job.h * SCALE, 3)
        for ((t, p) in job.tiles.zip(parts)) out.paste(p, t.x0 * SCALE, t.y0 * SCALE)
        if (outscale == SCALE.toDouble()) return out
        return ImageOps.resize(out, max(1, (job.w * outscale).toInt()), max(1, (job.h * outscale).toInt()), ImageOps.Interp.LANCZOS4)
    }

    fun enhance(img: Img, outscale: Double = SCALE.toDouble()): Img = prepare(img).let { assemble(it, infer(it), outscale) }
}
