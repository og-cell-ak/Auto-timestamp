package com.timestampgenius.app

import android.Manifest
import android.app.*
import android.content.*
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.pdf.PdfRenderer
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.*
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.*
import android.provider.MediaStore
import android.provider.Settings
import android.view.*
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color as ComposeColor
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs
import kotlin.math.max
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean

data class ScriptLine(val index: Int, val text: String, var timestampMs: Long? = null, var detected: Boolean = false)
data class WordBox(val text: String, val bounds: RectF)
data class OcrFrame(val text: String, val words: List<WordBox>, val lines: List<WordBox>)
data class LineLayout(var heightFraction: Float = .2f, var widthFraction: Float = 1f, var cornerRadius: Float = 10f, var unlocked: Boolean = false)
data class OverlayLayout(
    var x: Int = 24, var y: Int = 170, var width: Int = 720, var height: Int = 640,
    var lineCount: Int = 5, var scrollSpeed: Int = 0,
    var lines: MutableList<LineLayout> = MutableList(5) { LineLayout() }
)

object TextNorm {
    fun tokens(s: String) = s.lowercase().replace(Regex("[^\\p{L}\\p{N}']+"), " ")
        .split(" ").map { it.trim('\'') }.filter { it.isNotBlank() }
}

class SessionStore(private val context: Context) {
    private val p = context.getSharedPreferences("tg", Context.MODE_PRIVATE)
    fun loadScript(): List<String> = runCatching {
        val a = JSONArray(p.getString("script", "[]")); List(a.length()) { a.getString(it) }
    }.getOrDefault(emptyList())
    fun saveScript(lines: List<String>) = p.edit().putString("script", JSONArray(lines).toString()).apply()
    fun saveLines(lines: List<ScriptLine>) {
        val a = JSONArray()
        lines.forEach { a.put(JSONObject().apply {
            put("i", it.index); put("t", it.text); put("ms", it.timestampMs ?: -1); put("d", it.detected)
        }) }
        p.edit().putString("session", a.toString()).apply()
    }
    fun loadLines(): MutableList<ScriptLine> = runCatching {
        val a = JSONArray(p.getString("session", "[]")); MutableList(a.length()) {
            val o=a.getJSONObject(it); ScriptLine(o.getInt("i"),o.getString("t"),o.getLong("ms").takeIf { v -> v >= 0 },o.getBoolean("d"))
        }
    }.getOrDefault(mutableListOf())
    fun setLastPdf(uri: Uri) = p.edit().putString("lastUri", uri.toString()).apply()
    fun lastPdf(): Uri? = p.getString("lastUri", null)?.let(Uri::parse)
    fun clearSession() = p.edit().remove("script").remove("session").remove("lastUri").apply()
    fun layout(): OverlayLayout {
        val raw=p.getString("layout",null) ?: return OverlayLayout()
        return runCatching {
            val o=JSONObject(raw); val n=o.optInt("n",5).coerceIn(1,20); val a=o.optJSONArray("l")
            OverlayLayout(o.optInt("x",24),o.optInt("y",170),o.optInt("w",720),o.optInt("h",640),n,o.optInt("s",0),
                MutableList(n){i->val q=a?.optJSONObject(i);LineLayout(q?.optDouble("h",1.0/n)?.toFloat()?:1f/n,q?.optDouble("w",1.0)?.toFloat()?:1f,q?.optDouble("r",10.0)?.toFloat()?:10f,q?.optBoolean("u",false)?:false)})
        }.getOrDefault(OverlayLayout())
    }
    fun saveLayout(l: OverlayLayout) {
        p.edit().putString("layout",JSONObject().apply {
            put("x",l.x);put("y",l.y);put("w",l.width);put("h",l.height);put("n",l.lineCount);put("s",l.scrollSpeed)
            put("l",JSONArray().apply { l.lines.forEach { q->put(JSONObject().apply{put("h",q.heightFraction);put("w",q.widthFraction);put("r",q.cornerRadius);put("u",q.unlocked)}) } })
        }.toString()).apply()
    }
}

class FuzzyMatcher(private val lines: List<String>) {
    var current = 0; private var observed = mutableListOf<String>(); var progress = 0
    fun feed(s: String): Boolean {
        if (current >= lines.size) return false
        val expected=TextNorm.tokens(lines[current]); if(expected.isEmpty()){current++;return true}
        observed += TextNorm.tokens(s); if(observed.size>expected.size*4+20) observed=observed.takeLast(expected.size*4+20).toMutableList()
        var p=0
        for (o in observed) if(p<expected.size && similar(expected[p],o)) p++
        val finished=p>=expected.size; progress=p
        if(finished){current++;observed.clear();progress=0}
        return finished
    }
    private fun similar(a:String,b:String):Boolean {
        if(a==b)return true
        if(a.length<3||b.length<3)return a.firstOrNull()==b.firstOrNull()
        if(a.first()!=b.first())return false
        return abs(a.length-b.length)<=2 || distance(a,b)<=max(1,minOf(a.length,b.length)/3)
    }
    private fun distance(a:String,b:String):Int {
        val d=IntArray(b.length+1){it}
        for(i in 1..a.length){var prev=d[0];d[0]=i;for(j in 1..b.length){val t=d[j];d[j]=minOf(d[j]+1,d[j-1]+1,prev+if(a[i-1]==b[j-1])0 else 1);prev=t}}
        return d[b.length]
    }
}

class VoskEngine(private val context: Context, private val assetName: String):AutoCloseable {
    private var model:Model?=null; private var r:Recognizer?=null
    fun start():Boolean=runCatching {
        val dir=File(context.filesDir,assetName)
        if(!File(dir,".ready").exists()){ if(dir.exists())dir.deleteRecursively();dir.mkdirs();copyAssets(context,assetName,dir);File(dir,".ready").writeText("ok") }
        model=Model(dir.absolutePath);r=Recognizer(model,16000f);true
    }.getOrDefault(false)
    fun accept(data:ByteArray,n:Int):String?=runCatching { val rr=r?:return null; if(rr.acceptWaveForm(data,n))JSONObject(rr.result).optString("text") else JSONObject(rr.partialResult).optString("partial") }.getOrNull()
    override fun close(){runCatching{r?.close()};runCatching{model?.close()};r=null;model=null}
    private fun copyAssets(c:Context,path:String,to:File){
        val list=c.assets.list(path) ?: emptyArray()
        if(list.isEmpty()){c.assets.open(path).use{ i->to.outputStream().use{i.copyTo(it)}};return}
        list.forEach{copyAssets(c,path+"/"+it,File(to,it))}
    }
}

class OcrEngine:AutoCloseable {
    private val latin=TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val dev=TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build())
    suspend fun read(bitmap:Bitmap):OcrFrame {
        val input=InputImage.fromBitmap(bitmap,0)
        val a=await(latin.process(input));val b=runCatching{await(dev.process(input))}.getOrNull()
        val t=if((b?.text?.length?:0)>(a?.text?.length?:0))b else a
        if(t==null)return OcrFrame("",emptyList(),emptyList())
        val words=mutableListOf<WordBox>();val lines=mutableListOf<WordBox>()
        t.textBlocks.forEach{blk->blk.lines.forEach{line->line.elements.forEach{e->e.boundingBox?.let{words+=WordBox(e.text,RectF(it))}};line.boundingBox?.let{lines+=WordBox(line.text,RectF(it))}}}
        return OcrFrame(t.text,words,lines)
    }
    override fun close(){latin.close();dev.close()}
    private suspend fun <T>await(task:com.google.android.gms.tasks.Task<T>):T=suspendCancellableCoroutine{c->task.addOnSuccessListener{c.resume(it)}.addOnFailureListener{c.resumeWithException(it)}}
}

object PdfScriptParser {
    suspend fun parse(c:Context,uri:Uri):Result<List<String>> = runCatching {
        val fd=c.contentResolver.openFileDescriptor(uri,"r")?:error("Could not open PDF.")
        fd.use { pfd -> val r=PdfRenderer(pfd); r.use {renderer->
            val o=OcrEngine(); try {
                val out=mutableListOf<String>()
                for(pi in 0 until renderer.pageCount){
                    val page=renderer.openPage(pi)
                    try {
                        val scale=1.8f; val bmp=Bitmap.createBitmap((page.width*scale).toInt(),(page.height*scale).toInt(),Bitmap.Config.ARGB_8888)
                        bmp.eraseColor(Color.WHITE);page.render(bmp,null,null,PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                        val ys=yellowBands(bmp); if(ys.isNotEmpty()){
                            val frame=o.read(bmp); val cuts=regions(ys,bmp.height)
                            cuts.forEach{(top,bottom)->out+=frame.lines.filter{val y=(it.bounds.top+it.bounds.bottom)/2f;y in top..bottom}.joinToString(" "){it.text}.trim()}
                        }
                        bmp.recycle()
                    } finally {page.close()}
                }
                if(out.isEmpty()||out.all{it.isBlank()})error("No yellow separators were detected or no readable text was found.")
                out
            } finally{o.close()}
        }}
    }
    private fun yellowBands(b:Bitmap):List<Int>{
        val ys=mutableListOf<Int>();var on=false;var s=0;val step=max(1,b.width/250)
        for(y in 0 until b.height){var n=0;for(x in 0 until b.width step step){val c=b.getPixel(x,y);val rr=Color.red(c);val gg=Color.green(c);val bb=Color.blue(c);if(rr>180&&gg>140&&bb<130&&rr>bb*1.4f)n++};val yes=n>=8;if(yes&&!on){s=y;on=true};if(!yes&&on){if(y-s>=2)ys+=(s+y-1)/2;on=false}}
        if(on)ys+=(s+b.height-1)/2;return ys
    }
    private fun regions(ys:List<Int>,h:Int):List<Pair<Float,Float>>{val out=mutableListOf<Pair<Float,Float>>();var top=0f;ys.forEach{y->if(y-top>8)out+=top to (y-1f);top=y+2f};if(h-top>8)out+=top to (h-1f);return out}
}

class GuideView(c:Context):View(c){
    var m=OverlayLayout();var edit=false;var selected=0
    private val border=Paint(3).apply{style=Paint.Style.STROKE;strokeWidth=4f;color=Color.YELLOW}
    private val line=Paint(3).apply{style=Paint.Style.STROKE;strokeWidth=2f;color=Color.YELLOW}
    private val fill=Paint(3).apply{style=Paint.Style.FILL;color=0x18FFFF00}
    private val handle=Paint(3).apply{style=Paint.Style.FILL;color=Color.YELLOW}
    private var mode=0;private var lx=0f;private var ly=0f
    override fun onDraw(c:Canvas){val x=m.x.toFloat();val y=m.y.toFloat();val w=m.width.toFloat();val h=m.height.toFloat();c.drawRoundRect(x,y,x+w,y+h,12f,12f,fill);c.drawRoundRect(x,y,x+w,y+h,12f,12f,border);var cy=y;val total=m.lines.sumOf{it.heightFraction.coerceAtLeast(.04f).toDouble()}.toFloat();m.lines.forEachIndexed{i,q->val hh=h*(q.heightFraction.coerceAtLeast(.04f)/total);val rect=RectF(x,cy,x+w*q.widthFraction.coerceIn(.35f,1f),cy+hh);c.drawRoundRect(rect,q.cornerRadius,q.cornerRadius,line);if(edit&&i==selected)c.drawCircle(rect.right,rect.bottom,12f,handle);cy+=hh}}
    override fun onTouchEvent(e:MotionEvent):Boolean{if(!edit)return false;when(e.actionMasked){MotionEvent.ACTION_DOWN->{lx=e.rawX;ly=e.rawY;mode=hit(e.rawX,e.rawY);return true};MotionEvent.ACTION_MOVE->{val dx=e.rawX-lx;val dy=e.rawY-ly;lx=e.rawX;ly=e.rawY;when(mode){1->{m.x+=dx.toInt();m.y+=dy.toInt()};2->{m.width=max(280,m.width+dx.toInt())};3->{m.height=max(180,m.height+dy.toInt())};4->{val q=m.lines[selected];q.heightFraction=(q.heightFraction+dy/1200f).coerceIn(.04f,.8f);q.widthFraction=(q.widthFraction+dx/1600f).coerceIn(.35f,1f)}};invalidate();return true}};return true}
    private fun hit(px:Float,py:Float):Int{val x=m.x.toFloat();val y=m.y.toFloat();val r=x+m.width;val b=y+m.height;if(px in x..r&&py in y..b){var cy=y;val total=m.lines.sumOf{it.heightFraction.coerceAtLeast(.04f).toDouble()}.toFloat();m.lines.forEachIndexed{i,q->{val hh=m.height*(q.heightFraction.coerceAtLeast(.04f)/total);if(py in cy..cy+hh){selected=i;return 4};cy+=hh}};return 1};return 0}
}

class HighlightView(c:Context):View(c){var rects:List<RectF> = emptyList();private val p=Paint(3).apply{style=Paint.Style.FILL;color=0xAAFFD600.toInt()};override fun onDraw(c:Canvas){rects.forEach{c.drawRoundRect(it,5f,5f,p)}}}

class TimestampService:Service(){
    companion object{const val PREPARE="tg.prepare";const val START="tg.start";const val STOP="tg.stop";const val SAVE="tg.save";const val SET="tg.set";const val CODE="code";const val DATA="data";const val NOTIF=991}
    private lateinit var store:SessionStore;private lateinit var wm:WindowManager;private val main=Handler(Looper.getMainLooper())
    private var icon:TextView?=null;private var iconP:WindowManager.LayoutParams?=null;private var menu:LinearLayout?=null;private var guide:GuideView?=null;private var hi:HighlightView?=null
    private var projection:MediaProjection?=null;private var reader:ImageReader?=null;private var display:VirtualDisplay?=null
    private var audio:AudioRecord?=null;private var audioThread:Thread?=null;private var ocr:OcrEngine?=null
    private var engineHi:VoskEngine?=null;private var engineEn:VoskEngine?=null;private var matcher:FuzzyMatcher?=null
    private var lines=mutableListOf<ScriptLine>();private val recording=AtomicBoolean(false);private var startNs=0L;private var visibleWords=emptyList<WordBox>()
    override fun onCreate(){super.onCreate();store=SessionStore(this);wm=getSystemService(WINDOW_SERVICE) as WindowManager;notificationChannel();lines=store.loadScript().mapIndexed{i,s->ScriptLine(i,s)}.toMutableList();showIcon()}
    override fun onStartCommand(i:Intent?,f:Int,id:Int):Int{when(i?.action){PREPARE->prepare(i);START->startRecording();STOP->stopRecording();SAVE->savePdf();SET->setLines()};return START_STICKY}
    private fun prepare(i:Intent){if(projection!=null)return;val code=i.getIntExtra(CODE,Activity.RESULT_CANCELED);val data=if(Build.VERSION.SDK_INT>=33)i.getParcelableExtra(DATA,Intent::class.java) else @Suppress("DEPRECATION") i.getParcelableExtra(DATA) ?: return;val mgr=getSystemService(MediaProjectionManager::class.java);projection=mgr.getMediaProjection(code,data);startForeground(NOTIF,notification("Ready"),ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);setupCapture()}
    private fun setupCapture(){val p=projection?:return;val dm=resources.displayMetrics;reader=ImageReader.newInstance(dm.widthPixels,dm.heightPixels,PixelFormat.RGBA_8888,2);reader!!.setOnImageAvailableListener({capture(it) },main);display=p.createVirtualDisplay("TimestampGenius",dm.widthPixels,dm.heightPixels,dm.densityDpi,DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,reader!!.surface,null,main);ocr=OcrEngine()}
    private fun showIcon(){if(icon?.parent!=null)return;val v=TextView(this).apply{text="TG";textSize=13f;gravity=Gravity.CENTER;setTextColor(Color.BLACK);setBackgroundColor(Color.YELLOW);setOnTouchListener(DragTouch());setOnClickListener{toggleMenu()}};icon=v;val p=WindowManager.LayoutParams(64,64,WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.START;x=16;y=100};iconP=p;runCatching{wm.addView(v,p)}}
    private fun toggleMenu(){if(menu?.parent!=null){remove(menu);return};val l=LinearLayout(this).apply{orientation=LinearLayout.VERTICAL;setPadding(8,8,8,8);setBackgroundColor(Color.WHITE);elevation=20f};btn(l,"START"){startRecording();remove(menu)};btn(l,"STOP"){stopRecording();remove(menu)};btn(l,"SAVE"){savePdf();remove(menu)};btn(l,"SET LINES"){setLines();remove(menu)};val p=WindowManager.LayoutParams(240,-2,WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.START;x=(iconP?.x?:16)+72;y=iconP?.y?:100};menu=l;runCatching{wm.addView(l,p)}}
    private fun setLines(){val gv=GuideView(this).apply{m=store.layout();edit=true};guide=gv;val gp=WindowManager.LayoutParams(-1,-1,WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.START};runCatching{wm.addView(gv,gp)};val p=LinearLayout(this).apply{orientation=LinearLayout.HORIZONTAL;setPadding(8,8,8,8);setBackgroundColor(Color.WHITE)};btn(p,"−"){adjust(gv,-1)};btn(p,"+"){adjust(gv,1)};btn(p,"LOCK"){toggleLock(gv)};btn(p,"CORNER +"){val q=gv.m.lines[gv.selected];q.cornerRadius=(q.cornerRadius+4).coerceAtMost(40f);gv.invalidate()};btn(p,"SCROLL "+gv.m.scrollSpeed){gv.m.scrollSpeed=(gv.m.scrollSpeed+1)%10;(p.getChildAt(4)as Button).text="SCROLL "+gv.m.scrollSpeed};btn(p,"SAVE LAYOUT"){store.saveLayout(gv.m);remove(gv);remove(p)};btn(p,"CLOSE"){remove(gv);remove(p)};val pp=WindowManager.LayoutParams(-2,-2,WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.CENTER_HORIZONTAL;y=10};runCatching{wm.addView(p,pp)}}
    private fun adjust(g:GuideView,d:Int){val n=(g.m.lineCount+d).coerceIn(1,20);g.m.lineCount=n;while(g.m.lines.size<n)g.m.lines.add(LineLayout());while(g.m.lines.size>n)g.m.lines.removeAt(g.m.lines.lastIndex);val q=1f/n;g.m.lines.forEach{if(!it.unlocked)it.heightFraction=q};g.invalidate()}
    private fun toggleLock(g:GuideView){val i=g.selected;g.m.lines[i].unlocked=!g.m.lines[i].unlocked;g.invalidate()}
    private fun startRecording(){if(recording.get())return;if(Build.VERSION.SDK_INT<29){toast("Device audio capture needs Android 10 or newer.");return};val p=projection?:run{toast("Press START on the main screen first.");return};lines=store.loadScript().mapIndexed{i,s->ScriptLine(i,s)}.toMutableList();if(lines.isEmpty())lines=store.loadLines();matcher=FuzzyMatcher(lines.map{it.text});startNs=System.nanoTime();recording.set(true);ensureRuntimeOverlays();val sr=16000;val minb=AudioRecord.getMinBufferSize(sr,AudioFormat.CHANNEL_IN_MONO,AudioFormat.ENCODING_PCM_16BIT).coerceAtLeast(8192);val cfg=AudioPlaybackCaptureConfiguration.Builder(p).addMatchingUsage(AudioAttributes.USAGE_MEDIA).addMatchingUsage(AudioAttributes.USAGE_GAME).addMatchingUsage(AudioAttributes.USAGE_UNKNOWN).build();audio=runCatching{AudioRecord.Builder().setAudioFormat(AudioFormat.Builder().setSampleRate(sr).setEncoding(AudioFormat.ENCODING_PCM_16BIT).setChannelMask(AudioFormat.CHANNEL_IN_MONO).build()).setBufferSizeInBytes(minb*2).setAudioPlaybackCaptureConfig(cfg).build()}.getOrNull();if(audio==null){recording.set(false);toast("The source app blocked playback capture.");return};engineHi=VoskEngine(this,"vosk-hi").takeIf{it.start()};engineEn=VoskEngine(this,"vosk-en").takeIf{it.start()};if(engineHi==null&&engineEn==null){recording.set(false);toast("Offline speech model could not start.");return};audio!!.startRecording();audioThread=Thread{val buf=ByteArray(minb);while(recording.get()){val n=runCatching{audio!!.read(buf,0,buf.size)}.getOrDefault(0);if(n>0){val hy=buildList{engineHi?.accept(buf,n)?.takeIf{it.isNotBlank()}?.let(::add);engineEn?.accept(buf,n)?.takeIf{it.isNotBlank()}?.let(::add)};val best=hy.maxByOrNull{score(lines.getOrNull(matcher?.current?:0)?.text.orEmpty(),it)};if(best!=null)handleSpeech(best)}}}.apply{start()} }
    private fun handleSpeech(s:String){val m=matcher?:return;val before=m.current;val finished=m.feed(s);val idx=before;if(finished&&idx in lines.indices){lines[idx].timestampMs=(System.nanoTime()-startNs)/1_000_000;lines[idx].detected=true;store.saveLines(lines);hi?.rects=emptyList()};val n=m.progress;main.post{hi?.rects=visibleWords.take(n).map{it.bounds};hi?.invalidate()};if(m.current>=lines.size)main.post{stopRecording()}}
    private fun score(expected:String,observed:String):Int{val e=TextNorm.tokens(expected);val o=TextNorm.tokens(observed);return o.count{ot->e.any{et->ot==et||(ot.length>2&&et.length>2&&ot.first()==et.first())}}}
    private fun ensureRuntimeOverlays(){if(guide?.parent==null){val g=GuideView(this).apply{m=store.layout();edit=false};guide=g;val p=WindowManager.LayoutParams(-1,-1,WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.START};runCatching{wm.addView(g,p)}};if(hi?.parent==null){val h=HighlightView(this);hi=h;val p=WindowManager.LayoutParams(-1,-1,WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,PixelFormat.TRANSLUCENT).apply{gravity=Gravity.TOP or Gravity.START};runCatching{wm.addView(h,p)}}}
    private fun capture(r:ImageReader){val img=r.acquireLatestImage()?:return;try{val b=imageBitmap(img)?:return;CoroutineScope(Dispatchers.Default).launch{val frame=runCatching{ocr?.read(b)}.getOrNull();if(frame!=null)main.post{visibleWords=frame.words};b.recycle()}}finally{img.close()}}
    private fun imageBitmap(i:Image):Bitmap?{val pl=i.planes.firstOrNull()?:return null;val ps=pl.pixelStride;val row=pl.rowStride;val pad=row-ps*i.width;val tmp=Bitmap.createBitmap(i.width+pad/ps,i.height,Bitmap.Config.ARGB_8888);pl.buffer.rewind();tmp.copyPixelsFromBuffer(pl.buffer);return if(pad==0)tmp else Bitmap.createBitmap(tmp,0,0,i.width,i.height).also{tmp.recycle()}}
    private fun stopRecording(){recording.set(false);runCatching{audio?.stop()};audio?.release();audio=null;engineHi?.close();engineHi=null;engineEn?.close();engineEn=null;store.saveLines(lines);toast("Stopped. Timestamps are retained.")}
    private fun savePdf(){if(recording.get())stopRecording();val out=if(lines.isNotEmpty())lines else store.loadLines();if(out.isEmpty()){toast("No timestamps recorded yet.");return};val name="ScriptTimestamps_"+SimpleDateFormat("yyyy-MM-dd_HH-mm",Locale.US).format(Date())+".pdf";val v=ContentValues().apply{put(MediaStore.Downloads.DISPLAY_NAME,name);put(MediaStore.Downloads.MIME_TYPE,"application/pdf");put(MediaStore.Downloads.RELATIVE_PATH,"Download/ScriptTimestamper")};val uri=contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI,v)?:run{toast("Could not create the PDF.");return};runCatching{contentResolver.openOutputStream(uri)!!.use{PdfWriter.write(out,it)};store.setLastPdf(uri);toast("Saved "+name+" in Downloads/ScriptTimestamper")}.onFailure{contentResolver.delete(uri,null,null);toast("PDF save failed: "+(it.message?: "unknown error"))}}
    private fun btn(l:LinearLayout,s:String,click:()->Unit){l.addView(Button(this).apply{text=s;setTextColor(Color.BLACK);setBackgroundColor(Color.WHITE);setOnClickListener{click()}})}
    private fun remove(v:View?){if(v!=null&&v.parent!=null)runCatching{wm.removeView(v)}}
    private fun toast(s:String){main.post{Toast.makeText(this,s,Toast.LENGTH_LONG).show()}}
    private fun notificationChannel(){if(Build.VERSION.SDK_INT>=26)getSystemService(NotificationManager::class.java).createNotificationChannel(NotificationChannel("tg","Timestamp Genius",NotificationManager.IMPORTANCE_LOW))}
    private fun notification(s:String)=androidx.core.app.NotificationCompat.Builder(this,"tg").setSmallIcon(android.R.drawable.ic_menu_recent_history).setContentTitle("Timestamp Genius").setContentText(s).setOngoing(true).build()
    private inner class DragTouch:View.OnTouchListener{var x=0f;var y=0f;var moved=false;override fun onTouch(v:View,e:MotionEvent):Boolean{val p=iconP?:return false;when(e.actionMasked){MotionEvent.ACTION_DOWN->{x=e.rawX;y=e.rawY;moved=false;return false};MotionEvent.ACTION_MOVE->{val dx=e.rawX-x;val dy=e.rawY-y;if(abs(dx)>4||abs(dy)>4)moved=true;p.x+=dx.toInt();p.y=max(0,p.y+dy.toInt());runCatching{wm.updateViewLayout(icon,p)};x=e.rawX;y=e.rawY;return true};MotionEvent.ACTION_UP->return moved};return false}}
    override fun onDestroy(){recording.set(false);runCatching{audio?.release()};runCatching{engineHi?.close()};runCatching{engineEn?.close()};runCatching{display?.release()};runCatching{reader?.close()};runCatching{projection?.stop()};remove(icon);remove(menu);remove(guide);remove(hi);main.removeCallbacksAndMessages(null);super.onDestroy()}
}

object PdfWriter {
    fun write(lines:List<ScriptLine>,out:OutputStream){
        val doc=android.graphics.pdf.PdfDocument();val pw=595;val ph=842;val ml=40f;val tw=120f;val textPaint=Paint(3).apply{color=Color.BLACK;textSize=11f};val timePaint=Paint(textPaint).apply{typeface=Typeface.create(Typeface.DEFAULT,Typeface.BOLD);textSize=10f}
        var pageNo=1;var page:android.graphics.pdf.PdfDocument.Page?=null;var c:Canvas?=null;var y=ml
        fun newPage(){page=doc.startPage(android.graphics.pdf.PdfDocument.PageInfo.Builder(pw,ph,pageNo++).create());c=page!!.canvas;y=ml}
        fun finish(){page?.let{doc.finishPage(it)};page=null;c=null}
        fun wrap(t:String,maxWidth:Float):List<String>{val words=t.split(Regex("\\s+"));val arr=mutableListOf<String>();var cur="";for(w in words){val next=if(cur.isBlank())w else cur+" "+w;if(textPaint.measureText(next)<=maxWidth)cur=next else{if(cur.isNotBlank())arr+=cur;cur=w}};if(cur.isNotBlank())arr+=cur;return if(arr.isEmpty())listOf("")else arr}
        newPage()
        for(line in lines){val rows=wrap(line.text,pw-ml-tw);val h=max(22f,rows.size*16f+6);if(y+h>ph-ml){finish();newPage()};c!!.drawText(line.timestampMs?.let{fmt(it)}?:"--:--:--.---",ml,y+11,timePaint);rows.forEachIndexed{i,s->c!!.drawText(s,ml+tw,y+11+i*16,textPaint)};y+=h+5}
        finish();doc.writeTo(out);doc.close()
    }
    private fun fmt(ms:Long):String{val h=ms/3600000;val m=(ms%3600000)/60000;val s=(ms%60000)/1000;val z=ms%1000;return String.format(Locale.US,"%02d:%02d:%02d.%03d",h,m,s,z)}
}

class MainActivity:ComponentActivity(){
    private val store by lazy{SessionStore(this)}
    private val notifyLauncher=registerForActivityResult(ActivityResultContracts.RequestPermission()){}
    private val projectionLauncher=registerForActivityResult(ActivityResultContracts.StartActivityForResult()){r->if(r.resultCode==RESULT_OK&&r.data!=null){val i=Intent(this,TimestampService::class.java).apply{action=TimestampService.PREPARE;putExtra(TimestampService.CODE,r.resultCode);putExtra(TimestampService.DATA,r.data)};ContextCompat.startForegroundService(this,i)}}
    override fun onCreate(b:Bundle?){super.onCreate(b);setContent{AppUi()}}
    @Composable private fun AppUi(){
        val scope=rememberCoroutineScope();var script by remember{mutableStateOf(store.loadScript())};var msg by remember{mutableStateOf<String?>(null)};var confirm by remember{mutableStateOf(false)}
        val picker=rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()){uri->if(uri!=null)scope.launch(Dispatchers.IO){val r=PdfScriptParser.parse(this@MainActivity,uri);withContext(Dispatchers.Main){r.onSuccess{script=it;store.saveScript(it);msg="PDF loaded: "+it.size+" timestamp lines detected."}.onFailure{msg=it.message?: "Could not read PDF."}}}}
        fun start(){if(Build.VERSION.SDK_INT>=23&&!Settings.canDrawOverlays(this@MainActivity)){startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,Uri.parse("package:"+packageName)));msg="Allow Display over other apps, then press START again.";return};if(Build.VERSION.SDK_INT>=33)notifyLauncher.launch(Manifest.permission.POST_NOTIFICATIONS);val m=getSystemService(MediaProjectionManager::class.java);projectionLauncher.launch(m.createScreenCaptureIntent())}
        Column(Modifier.fillMaxSize().padding(24.dp),verticalArrangement=Arrangement.spacedBy(16.dp)){
            Text("Timestamp Genius",style=MaterialTheme.typography.headlineMedium,color=ComposeColor.Black)
            Text("Script-aware timestamps from device audio + screen/PDF",color=ComposeColor.DarkGray)
            Button({picker.launch(arrayOf("application/pdf"))},Modifier.fillMaxWidth()){Text("UPLOAD PDF")}
            Button({start()},Modifier.fillMaxWidth()){Text("START")}
            Button({store.lastPdf()?.let{runCatching{startActivity(Intent(Intent.ACTION_VIEW,it).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))}.onFailure{msg="Cannot open the last PDF."}}?:run{msg="No PDF recorded yet"}},Modifier.fillMaxWidth()){Text("LAST PDF RECORDED")}
            Button({confirm=true},Modifier.fillMaxWidth()){Text("NEW SESSION")}
            if(script.isNotEmpty()){Text("Loaded script: "+script.size+" lines",style=MaterialTheme.typography.titleMedium);LazyColumn(Modifier.weight(1f)){items(script){Text(it,color=ComposeColor.Black)}}}else Spacer(Modifier.weight(1f))
            msg?.let{Text(it,color=ComposeColor.Black)}
        }
        if(confirm)AlertDialog(onDismissRequest={confirm=false},confirmButton={TextButton({store.clearSession();script=emptyList();msg="New session created.";confirm=false}){Text("Continue")}},dismissButton={TextButton({confirm=false}){Text("Cancel")}},title={Text("Clear current session?")},text={Text("This clears current script, timestamps and PDF reference. Saved files in Downloads are not deleted.")}})
    }
}
