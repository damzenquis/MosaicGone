package com.mosaicgone.app

import com.mosaicgone.app.ProcessingService.StartAction
import org.junit.Assert.assertEquals
import org.junit.Test

class ProcessingServiceTest {

    @Test
    fun startAction() {
        assertEquals(StartAction.START_JOB, ProcessingService.startAction(hasRequest = true, jobActive = false))
        // a start while a job is still running (or winding down) is acknowledged, never dropped silently
        assertEquals(StartAction.ALREADY_RUNNING, ProcessingService.startAction(hasRequest = true, jobActive = true))
        // restarted by the system with nothing to run
        assertEquals(StartAction.NOTHING_TO_RUN, ProcessingService.startAction(hasRequest = false, jobActive = false))
        assertEquals(StartAction.NOTHING_TO_RUN, ProcessingService.startAction(hasRequest = false, jobActive = true))
    }
}
