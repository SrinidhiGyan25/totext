import os
import tempfile
import subprocess
import zipfile
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from faster_whisper import WhisperModel

# ---------- Config (transcription) ----------
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")  # good on CPU
THREADS = int(os.getenv("WHISPER_THREADS", "4"))

RTF_GUESSES = {
    "tiny": 0.4,
    "base": 0.8,
    "small": 1.2,
    "medium": 2.0,
    "large-v3": 3.0,
}

_model = None
def get_model():
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, compute_type=COMPUTE_TYPE, cpu_threads=THREADS)
    return _model

def run_ffprobe_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def sanitize_filename(name: str) -> str:
    keep = "-_(). "
    return "".join(c for c in name if c.isalnum() or c in keep).strip() or "audio"

# ---------- Optional Summarization (per-request API key) ----------
def summarize_to_bullets(text: str, api_key: str, model: str = None) -> str:
    if not api_key:
        raise RuntimeError("No API key provided for summarization.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system = (
        "You are a meticulous note-taker. Summarize transcripts into clear, "
        "hierarchical bullet points without omitting important facts. Keep bullets short, "
        "group logically, follow chronology when helpful, and do not invent details."
    )
    user = (
        "Summarize the following transcript into thorough bullet points. "
        "Capture all key ideas, decisions, numbers, and action items. "
        "Use nested bullets when needed.\n\n"
        f"--- TRANSCRIPT START ---\n{text}\n--- TRANSCRIPT END ---"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ---------- App ----------
app = FastAPI(title="Audio → Text Plus", version="1.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audio → Text (.txt) + Optional Summary</title>
  <link rel="icon" href="/static/favicon.ico" />
  <style>
    :root { --bg:#0f172a; --card:#111827; --muted:#9ca3af; --text:#e5e7eb; --accent:#22d3ee; }
    html,body { background: radial-gradient(1200px 600px at 20% -10%, rgba(34,211,238,.15), transparent),
                               linear-gradient(180deg, #0b1020, #0f172a);
                height:100%; margin:0; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Inter', sans-serif; color: var(--text); }
    .shell { max-width: 900px; margin: 48px auto; padding: 0 16px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px; padding: 22px 22px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
    h1 { margin: 4px 0 12px; font-size: 28px; letter-spacing: .3px; }
    .muted { color: var(--muted); }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    .filebox { padding: 16px; border:1px dashed rgba(255,255,255,.18); border-radius: 14px; background: rgba(255,255,255,.02); }
    .select, input[type=file], input[type=password], input[type=text] {
      padding: .55rem .7rem; border-radius: 10px; border:1px solid rgba(255,255,255,.15); color: var(--text); background: #0b1224;
    }
    .w100 { flex:1; min-width: 240px; }
    button { padding:.75rem 1.1rem; border-radius: 12px; border:0; cursor:pointer; font-weight:600; }
    button.primary { background: linear-gradient(90deg, #06b6d4, #22d3ee); color:#02131c; }
    button.ghost { background: #0b1224; color: var(--text); border:1px solid rgba(255,255,255,.14); }
    .est { background: rgba(34,211,238,.07); padding: 0.85rem; border-radius: 12px; border: 1px solid rgba(34,211,238,.25); }
    progress { width: 100%; height: 14px; background:#0b1224; border-radius:8px; }
    .hidden { display:none; }
    .right { margin-left:auto; }
    .check { display:flex; align-items:center; gap:.5rem; }
    .hint { font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="shell">
    <div class="card">
      <h1>Audio → Text (.txt)</h1>
      <p class="muted">Upload audio. Get ETA. Download transcript. Add a bullet summary with your own OpenAI API key (optional).</p>

      <div class="filebox">
        <div class="row">
          <label for="file"><b>Audio file</b></label>
          <input id="file" class="w100" type="file" accept="audio/*" />
        </div>
        <div class="row" style="margin-top: .75rem;">
          <label for="model"><b>Model</b></label>
          <select id="model" class="select">
            <option value="tiny">tiny (fastest)</option>
            <option value="base" selected>base (balanced)</option>
            <option value="small">small (better quality)</option>
            <option value="medium">medium (slower)</option>
            <option value="large-v3">large-v3 (best quality)</option>
          </select>
          <label class="check">
            <input id="summarize" type="checkbox" />
            Summarize with ChatGPT
          </label>
        </div>
        <div class="row" id="apikeyrow" style="margin-top:.75rem; display:none;">
          <label for="apikey"><b>OpenAI API key</b></label>
          <input id="apikey" class="w100" type="password" placeholder="sk-... (kept only in this request)" autocomplete="off" />
        </div>
        <p class="hint" id="apikeyhint" style="display:none;">Your key is sent only with this request and not stored on the server.</p>
        <div style="margin-top: 0.75rem;">
          <button id="estimateBtn" class="ghost">Estimate time</button>
        </div>
      </div>

      <div id="estimateBox" class="est hidden" style="margin-top: 1rem;"></div>

      <div class="row" style="margin-top: 1rem;">
        <button id="startBtn" class="primary" disabled>Start transcription</button>
        <button id="resetBtn" class="ghost">Reset</button>
      </div>

      <div id="status" class="muted" style="margin-top: 1rem;"></div>

      <div id="progressWrap" class="hidden" style="margin-top: 0.75rem;">
        <progress id="progress" max="100" value="0"></progress>
      </div>
    </div>
  </div>

<script>
  const fileInput = document.getElementById('file');
  const modelSelect = document.getElementById('model');
  const estimateBtn = document.getElementById('estimateBtn');
  const startBtn = document.getElementById('startBtn');
  const resetBtn = document.getElementById('resetBtn');
  const estimateBox = document.getElementById('estimateBox');
  const statusEl = document.getElementById('status');
  const progressWrap = document.getElementById('progressWrap');
  const progressEl = document.getElementById('progress');
  const summarizeCb = document.getElementById('summarize');
  const apikeyRow = document.getElementById('apikeyrow');
  const apikeyHint = document.getElementById('apikeyhint');
  const apikeyInput = document.getElementById('apikey');

  function fmt(sec) {
    if (!sec || sec <= 0) return "a few seconds";
    const m = Math.floor(sec / 60);
    const s = Math.round(sec % 60);
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
  }

  summarizeCb.onchange = () => {
    const on = summarizeCb.checked;
    apikeyRow.style.display = on ? 'flex' : 'none';
    apikeyHint.style.display = on ? 'block' : 'none';
  };

  resetBtn.onclick = () => {
    fileInput.value = "";
    apikeyInput.value = "";
    estimateBox.classList.add('hidden');
    estimateBox.textContent = "";
    startBtn.disabled = true;
    statusEl.textContent = "";
    progressWrap.classList.add('hidden');
    progressEl.value = 0;
    summarizeCb.checked = false;
    apikeyRow.style.display = 'none';
    apikeyHint.style.display = 'none';
  };

  estimateBtn.onclick = async () => {
    const f = fileInput.files?.[0];
    if (!f) { alert("Please choose an audio file first."); return; }
    const fd = new FormData();
    fd.append("file", f);
    fd.append("model", modelSelect.value);
    statusEl.textContent = "Estimating…";

    const res = await fetch('/estimate', { method: 'POST', body: fd });
    if (!res.ok) {
      statusEl.textContent = "Estimation failed. Please try again.";
      return;
    }
    const data = await res.json();
    const dur = data.duration_seconds || 0;
    const est = data.estimated_time_seconds || 0;
    estimateBox.innerHTML = `Estimated processing time: <b>${fmt(est)}</b> for audio duration <b>${fmt(dur)}</b>. This is a rough estimate.`;
    estimateBox.classList.remove('hidden');
    startBtn.disabled = false;
    statusEl.textContent = "Ready to transcribe.";
  };

  startBtn.onclick = async () => {
    const f = fileInput.files?.[0];
    if (!f) { alert("Please choose an audio file first."); return; }
    if (summarizeCb.checked && !apikeyInput.value) {
      const ok = confirm("You checked 'Summarize with ChatGPT' but no API key is entered. Continue without summary?");
      if (!ok) return;
    }
    startBtn.disabled = true;
    statusEl.textContent = "Uploading & transcribing…";
    progressWrap.classList.remove('hidden');
    progressEl.value = 12;

    const fd = new FormData();
    fd.append("file", f);
    fd.append("model", modelSelect.value);
    fd.append("summarize", summarizeCb.checked ? "true" : "false");
    if (apikeyInput.value) fd.append("openai_api_key", apikeyInput.value);

    const res = await fetch('/transcribe', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.text();
      statusEl.textContent = "Transcription failed: " + err;
      progressEl.value = 0;
      return;
    }
    progressEl.value = 90;
    const blob = await res.blob();

    const base = f.name.replace(/\.[^/.]+$/, '') || "audio";
    const filename = (summarizeCb.checked && apikeyInput.value)
      ? `${base}_transcript_and_summary.zip`
      : `${base}.txt`;

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    statusEl.textContent = "Done! Your file has been downloaded.";
    progressEl.value = 100;
  };
</script>

</body>
</html>
    """
    return HTMLResponse(html)

@app.post("/estimate")
async def estimate_time(file: UploadFile = File(...), model: str = Form("base")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    duration = run_ffprobe_duration(tmp_path)
    os.unlink(tmp_path)

    model_key = model if model in RTF_GUESSES else MODEL_SIZE
    rtf = RTF_GUESSES.get(model_key, 1.0)
    est = duration * rtf

    return JSONResponse({
        "duration_seconds": duration,
        "estimated_time_seconds": est,
        "model": model_key,
        "rtf_used": rtf
    })

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("base"),
    summarize: bool = Form(False),
    openai_api_key: str = Form("", description="Optional per-request key for summarization"),
):
    tmp_dir = tempfile.mkdtemp(prefix="a2t_")
    src_path = os.path.join(tmp_dir, file.filename)
    with open(src_path, "wb") as f:
        f.write(await file.read())

    wav_path = os.path.join(tmp_dir, "input.wav")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", wav_path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return JSONResponse({"error": "ffmpeg failed to decode this file."}, status_code=400,
                            background=BackgroundTask(lambda: subprocess.run(["rm", "-rf", tmp_dir])))

    use_model = model if model in RTF_GUESSES else MODEL_SIZE
    m = get_model() if use_model == MODEL_SIZE else WhisperModel(use_model, compute_type=COMPUTE_TYPE, cpu_threads=THREADS)
    segments, info = m.transcribe(wav_path, beam_size=5, vad_filter=True)

    base = sanitize_filename(os.path.splitext(file.filename)[0])
    out_txt = os.path.join(tmp_dir, f"{base}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + " ")
    with open(out_txt, "r+", encoding="utf-8") as f:
        txt = f.read().strip() + "\n"
        f.seek(0); f.write(txt); f.truncate()

    if summarize and openai_api_key.strip():
        summary_txt = os.path.join(tmp_dir, f"{base}_summary.txt")
        try:
            full_text = open(out_txt, "r", encoding="utf-8").read()
            bullets = summarize_to_bullets(full_text, api_key=openai_api_key.strip())
            with open(summary_txt, "w", encoding="utf-8") as sf:
                sf.write(bullets + "\n")
        except Exception as e:
            with open(summary_txt, "w", encoding="utf-8") as sf:
                sf.write(f"[Summary failed: {e}]\n")

        zip_path = os.path.join(tmp_dir, f"{base}_transcript_and_summary.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_txt, arcname=f"{base}.txt")
            z.write(summary_txt, arcname=f"{base}_summary.txt")

        def cleanup():
            try: subprocess.run(["rm", "-rf", tmp_dir])
            except Exception: pass

        return FileResponse(zip_path, media_type="application/zip",
                            filename=f"{base}_transcript_and_summary.zip",
                            background=BackgroundTask(cleanup))

    def cleanup():
        try: subprocess.run(["rm", "-rf", tmp_dir])
        except Exception: pass

    return FileResponse(out_txt, media_type="text/plain",
                        filename=f"{base}.txt",
                        background=BackgroundTask(cleanup))

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
