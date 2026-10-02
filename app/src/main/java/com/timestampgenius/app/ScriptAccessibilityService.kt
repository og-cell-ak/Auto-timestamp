package com.timestampgenius.app

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

object AccessibilityBridge {
    @Volatile
    var service: ScriptAccessibilityService? = null

    fun scrollForward(): Boolean = service?.scrollForward() ?: false
}

class ScriptAccessibilityService : AccessibilityService() {
    override fun onServiceConnected() {
        super.onServiceConnected()
        AccessibilityBridge.service = this
        serviceInfo = serviceInfo.apply {
            eventTypes =
                AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED or
                    AccessibilityEvent.TYPE_VIEW_SCROLLED or
                    AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    fun scrollForward(): Boolean {
        val active = rootInActiveWindow
        if (active != null && scrollNode(active)) return true

        if (android.os.Build.VERSION.SDK_INT >= 21) {
            for (window in windows) {
                val root = window.root ?: continue
                if (scrollNode(root)) return true
            }
        }
        return false
    }

    private fun scrollNode(node: AccessibilityNodeInfo): Boolean {
        if (node.isScrollable &&
            node.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
        ) {
            return true
        }

        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            try {
                if (scrollNode(child)) return true
            } finally {
                child.recycle()
            }
        }
        return false
    }

    override fun onDestroy() {
        if (AccessibilityBridge.service === this) {
            AccessibilityBridge.service = null
        }
        super.onDestroy()
    }
}
