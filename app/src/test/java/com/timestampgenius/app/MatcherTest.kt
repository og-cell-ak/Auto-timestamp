package com.timestampgenius.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MatcherTest {
    @Test
    fun partialThenFinalCompletesLine() {
        val expected = mutableListOf("Hello world")
        val matcher = FuzzyMatcher { expected }

        val first = matcher.feed(SpeechResult("Hello", false))
        assertEquals(1, first.progress)
        assertEquals(-1, first.completedIndex)

        val second = matcher.feed(SpeechResult("world", true))
        assertEquals(0, second.completedIndex)
        assertEquals(1, matcher.current)
        assertEquals(0, matcher.progress)
    }

    @Test
    fun upcomingLineResynchronizesAndMarksSkippedLine() {
        val expected = mutableListOf("This line will be skipped", "Continue from here")
        val matcher = FuzzyMatcher { expected }

        val event = matcher.feed(SpeechResult("Continue from here", true))

        assertEquals(1, event.completedIndex)
        assertEquals(listOf(0), event.skippedIndices)
        assertEquals(2, matcher.current)
    }

    @Test
    fun normalizationHandlesMixedPunctuation() {
        val tokens = TextNorm.tokens("Hello, WORLD!")
        assertTrue(tokens.containsAll(listOf("hello", "world")))
    }
}
