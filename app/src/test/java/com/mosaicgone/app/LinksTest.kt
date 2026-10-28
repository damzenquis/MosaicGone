package com.mosaicgone.app

import org.junit.Assert.assertEquals
import org.junit.Test

class LinksTest {
    private fun links(text: String) = Links.find(text).map { text.substring(it) }

    @Test
    fun linksInBracketsAndSentences() {
        assertEquals(
            listOf("https://github.com/ladaapp"),
            links("The original Lada restorer model by ladaapp (https://github.com/ladaapp). About 2x slower"),
        )
        assertEquals(
            listOf("https://github.com/HypoX64/DeepMosaics"),
            links("• DeepMosaics by HypoX64 (https://github.com/HypoX64/DeepMosaics) (GPL-3.0) = Used to"),
        )
        assertEquals(
            listOf("https://github.com/damzenquis", "https://ko-fi.com/damzenquis"),
            links("View my projects at GitHub: https://github.com/damzenquis\nBuy me a coffee: https://ko-fi.com/damzenquis"),
        )
        assertEquals(listOf("https://github.com/ckkelvinchan/BasicVSR_PlusPlus"), links("(https://github.com/ckkelvinchan/BasicVSR_PlusPlus) (Apache-2.0)"))
        assertEquals(
            listOf("https://github.com/damzenquis/MosaicGone", "https://github.com/damzenquis"),
            links("• View the Source Code of this app at GitHub: https://github.com/damzenquis/MosaicGone\n\n• View my projects at GitHub: https://github.com/damzenquis"),
        )
        assertEquals(listOf("https://example.com/a"), links("see https://example.com/a."))
        assertEquals(emptyList<String>(), links("no links here, github.com without a scheme"))
    }

    /** every link in the app's texts is a full https:// link */
    @Test
    fun modelDescriptionLink() {
        val lada = ModelRegistry.find(com.mosaicgone.core.Options.LADA)!!
        assertEquals(listOf("https://github.com/ladaapp"), links(lada.description))
    }
}
