import java.net.HttpURLConnection
import java.net.URI
import java.util.zip.ZipInputStream

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val modelAssets = layout.buildDirectory.dir("generated/vosk-models")
android {
    namespace = "com.timestampgenius.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "com.timestampgenius.app"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }
    buildFeatures { compose = true }
    sourceSets["main"].assets.srcDir(modelAssets)
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    packaging { resources { excludes += "/META-INF/{AL2.0,LGPL2.1}" } }
}
dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("com.google.mlkit:text-recognition-devanagari:16.0.1")
    implementation("com.alphacephei:vosk-android:0.3.75")
    implementation("net.java.dev.jna:jna:5.18.1@aar")
}
val downloadModels = tasks.register("downloadVoskModels") {
    outputs.dir(modelAssets)
    doLast {
        val root = modelAssets.get().asFile
        root.mkdirs()
        val models = mapOf(
            "hi" to "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
            "en" to "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        )
        models.forEach { (key, url) ->
            val target = File(root, "vosk-" + key)
            if (File(target, ".ready").exists()) return@forEach
            val cache = File(layout.buildDirectory.get().asFile, "model-cache-" + key + ".zip")
            if (!cache.exists()) {
                val c = URI(url).toURL().openConnection() as HttpURLConnection
                c.connectTimeout = 30000; c.readTimeout = 180000
                c.setRequestProperty("User-Agent", "Timestamp-Genius")
                c.connect()
                check(c.responseCode in 200..299) { "Model download HTTP " + c.responseCode }
                c.inputStream.use { i -> cache.outputStream().use { o -> i.copyTo(o) } }
            }
            val temp = File(layout.buildDirectory.get().asFile, "model-tmp-" + key)
            if (temp.exists()) temp.deleteRecursively()
            temp.mkdirs()
            ZipInputStream(cache.inputStream().buffered()).use { z ->
                var e = z.nextEntry
                while (e != null) {
                    val d = File(temp, e.name)
                    if (e.isDirectory) d.mkdirs() else { d.parentFile.mkdirs(); d.outputStream().use { o -> z.copyTo(o) } }
                    e = z.nextEntry
                }
            }
            val src = temp.listFiles()?.firstOrNull { it.isDirectory } ?: error("Model archive empty")
            if (target.exists()) target.deleteRecursively()
            src.copyRecursively(target, overwrite = true)
            File(target, ".ready").writeText("ok")
        }
    }
}
tasks.named("preBuild").configure { dependsOn(downloadModels) }
