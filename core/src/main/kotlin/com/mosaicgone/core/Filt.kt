// Modified for MosaicGone, 2026: translated to Kotlin from util/filt.py of DeepMosaics
// (https://github.com/HypoX64/DeepMosaics, GPL-3.0); only the part the mosaic-removal pipeline uses.
// See THIRD_PARTY_NOTICES.md and third_party_licenses/deepmosaics/LICENSE.
package com.mosaicgone.core

/** Port of util/filt.py (only the part the mosaic-removal pipeline uses). */
object Filt {

    /** Zero-padded running median, like filt.medfilt. */
    fun medfilt(data: DoubleArray, window: Int): DoubleArray {
        require(window > 0 && window % 2 == 1) { "the medfilt window must be an odd number" }
        val pad = (window - 1) / 2
        val padded = DoubleArray(data.size + window - 1)
        System.arraycopy(data, 0, padded, pad, data.size)
        val buf = DoubleArray(window)
        return DoubleArray(data.size) { i ->
            System.arraycopy(padded, i, buf, 0, window)
            buf.sort()
            buf[pad]
        }
    }
}
