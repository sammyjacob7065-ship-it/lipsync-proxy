"""LipSync proxy, Wav2Lip (ONNX, CPU) edition, for a free Hugging Face Space.
Same API as the n8n lane expects:  POST /lipsync  ->  {job_id}   GET /status/<id>   GET /video/<id>   GET /health   (header x-api-key)
Only the speaker's face is changed, inside that line's time window. Narrator lines and listeners are never touched.
Output is VIDEO ONLY (assemble.py puts the voices on).  Wav2Lip licence: personal / research / non-commercial (check before monetising)."""
import os, uuid, threading, tempfile, shutil, subprocess, urllib.request, time
import numpy as np, cv2
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

KEY = os.environ.get("API_KEY", "change-me")
STUB = os.environ.get("LIPSYNC_STUB") == "1"          # test mode: fake face + fake model (no download)
MODEL_REPO, MODEL_FILE = "bluefoxcreation/Wav2lip-Onnx", "wav2lip_gan.onnx"
DEFAULT_FPS = int(os.environ.get("LIPSYNC_FPS", "12"))  # lower = faster on CPU; assemble.py resamples to 30
JOBS, LOCK, _SESS = {}, threading.Lock(), {}
app = FastAPI()

def auth(k):
    if k != KEY: raise HTTPException(401, "bad key")
def sh(*a): subprocess.run(a, check=True, capture_output=True)
def dl(url, path): urllib.request.urlretrieve(url, path)
def dur(p): return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).decode().strip())

# ---------- mel (Wav2Lip's own settings; numpy/scipy only, so it fits a 512 MB free host) ----------
def _mel_filters(sr=16000, n_fft=800, n_mels=80, fmin=55.0, fmax=7600.0):
    """same as librosa.filters.mel (Slaney scale and Slaney normalisation)"""
    def hz2mel(f):
        f = np.asarray(f, dtype=float); m = f / (200.0 / 3); lg = np.log(6.4) / 27.0
        return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) / lg, m)
    def mel2hz(m):
        m = np.asarray(m, dtype=float); lg = np.log(6.4) / 27.0
        return np.where(m >= 15.0, 1000.0 * np.exp(lg * (m - 15.0)), m * (200.0 / 3))
    fftf = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mf = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fdiff = np.diff(mf); ramps = mf[:, None] - fftf[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]; upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    return (w * (2.0 / (mf[2:n_mels + 2] - mf[:n_mels]))[:, None]).astype(np.float32)
_MELB = None
def mel_of(wav16k_path):
    global _MELB
    from scipy import signal
    from scipy.io import wavfile
    sr, y = wavfile.read(wav16k_path)
    y = y.astype(np.float32) / (32768.0 if y.dtype == np.int16 else 1.0)
    if y.ndim > 1: y = y.mean(1)
    y = signal.lfilter([1, -0.97], [1], y).astype(np.float32)
    n_fft, hop = 800, 200
    y = np.pad(y, n_fft // 2, mode="reflect")
    win = signal.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    nfr = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nfr)[:, None]
    mag = np.abs(np.fft.rfft(y[idx] * win[None, :], axis=1)).T          # (401, frames)
    if _MELB is None: _MELB = _mel_filters()
    S = 20 * np.log10(np.maximum(1e-5, _MELB @ mag)) - 20
    return np.clip(8 * ((S + 100) / 100) - 4, -4, 4).astype(np.float32)   # (80, frames), 80 frames per second

def session():
    if "s" not in _SESS:
        import onnxruntime as ort
        path = os.path.join(tempfile.gettempdir(), MODEL_FILE)
        if not os.path.exists(path):
            urllib.request.urlretrieve("https://huggingface.co/%s/resolve/main/%s" % (MODEL_REPO, MODEL_FILE), path + ".part"); os.replace(path + ".part", path)
        o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESS["s"] = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    return _SESS["s"]

def run_model(mel16, face96):
    """mel16: (80,16)  face96: (96,96,3) BGR uint8  ->  (96,96,3) BGR uint8"""
    if STUB:
        out = face96.copy(); out[60:80, 30:66] = (30, 30, 120) if int(mel16.mean() * 10) % 2 else (60, 60, 160); return out
    masked = face96.copy(); masked[48:] = 0
    vid = (np.concatenate([masked, face96], axis=2) / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
    m = mel16[None, None].astype(np.float32)
    pred = session().run(None, {"mel": m, "vid": vid})[0][0]
    return np.clip(pred.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

# ---------- faces ----------
_CASC = None
def faces_in(frame):
    """boxes (x,y,w,h), left to right"""
    global _CASC
    if STUB:
        h, w = frame.shape[:2]; return [(int(w * 0.30), int(h * 0.30), int(w * 0.28), int(w * 0.28))]
    if _CASC is None: _CASC = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = _CASC.detectMultiScale(g, 1.1, 5, minSize=(50, 50))
    return sorted([tuple(int(v) for v in b) for b in f], key=lambda b: b[0])

def paste(frame, box, mel16):
    """Replace the lower half of the speaker's face with the generated mouth area; the 10 rows above fade in softly."""
    x, y, w, h = box; pad = int(w * 0.25)
    x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    if x2 - x1 < 10 or y2 - y1 < 10: return frame
    crop = frame[y1:y2, x1:x2]; fh, fw = crop.shape[:2]
    gen = cv2.resize(run_model(mel16, cv2.resize(crop, (96, 96))), (fw, fh))
    res = crop.copy(); top = fh // 2
    res[top:] = gen[top:]
    for k in range(min(10, top)):
        a = k / 10.0; r = top - 10 + k
        if r >= 0: res[r] = cv2.addWeighted(crop[r], 1 - a, gen[r], a, 0)
    frame = frame.copy(); frame[y1:y2, x1:x2] = res
    return frame

# ---------- the job ----------
def work(jid, req):
    j = JOBS[jid]; d = tempfile.mkdtemp(); opt = req.get("options", {}); fps = int(opt.get("fps") or DEFAULT_FPS)
    try:
        with LOCK:                                        # one scene at a time: the CPU is small
            j["status"] = "running"
            src = os.path.join(d, "src.mp4")
            if req["source_type"] == "image":
                dl(req["source_url"], os.path.join(d, "s.png"))
                sh("ffmpeg", "-y", "-loop", "1", "-i", os.path.join(d, "s.png"), "-t", "1", "-r", str(fps), "-pix_fmt", "yuv420p", src)
            else: dl(req["source_url"], src)
            # speaker timeline from the REAL audio lengths (same order and no gaps, exactly like assemble.py joins them)
            t, wins, lst = 0.0, [], []
            for l in req["lines"]:
                p = os.path.join(d, "l%s.mp3" % l["line_index"]); dl(l["audio_url"], p)
                s, e = t, t + dur(p); t = e; lst.append(p)
                wins.append({"s": s, "e": e, "speaker": l["speaker"], "cid": l.get("character_id"), "talk": bool(l.get("is_dialogue"))})
            total = t
            j["timeline"] = [[w["speaker"], round(w["s"], 2), round(w["e"], 2)] for w in wins]
            lst_file = os.path.join(d, "list.txt"); open(lst_file, "w").write("".join("file '%s'\n" % p for p in lst))
            wav = os.path.join(d, "all.wav"); sh("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst_file, "-ar", "16000", "-ac", "1", wav)
            mel = mel_of(wav)
            base = os.path.join(d, "base.mp4")
            sh("ffmpeg", "-y", "-stream_loop", "-1", "-i", src, "-t", "%.3f" % total, "-an", "-r", str(fps), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", base)
            order = [c for c in (opt.get("face_order") or [c["character_id"] for c in req.get("characters", [])])]
            cap = cv2.VideoCapture(base); W, H = int(cap.get(3)), int(cap.get(4))
            raw = os.path.join(d, "raw.mp4"); vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            last, i, n = {}, 0, int(total * fps)
            while True:
                ok, fr = cap.read()
                if not ok or i >= n: break
                tt = i / fps; w = next((x for x in wins if x["s"] <= tt < x["e"]), None)
                if w and w["talk"]:
                    fb = faces_in(fr)
                    if fb:
                        idx = order.index(w["cid"]) if w["cid"] in order else 0
                        b = fb[min(idx, len(fb) - 1)] if len(fb) > 1 else fb[0]
                        key = idx; p = last.get(key)
                        last[key] = tuple(int(0.5 * a + 0.5 * c) for a, c in zip(b, p)) if p else b     # smooth the box over time
                    if last.get(idx if fb else 0) or last:
                        box = last.get(idx if fb else 0) or next(iter(last.values()))
                        st = int(tt * 80); chunk = mel[:, st:st + 16]
                        if chunk.shape[1] < 16: chunk = mel[:, -16:]
                        fr = paste(fr, box, chunk)
                vw.write(fr); i += 1; j["progress"] = "%d/%d frames" % (i, n)
            vw.release(); cap.release()
            final = os.path.join(tempfile.gettempdir(), jid + ".mp4")
            sh("ffmpeg", "-y", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", final)
            j.update(status="done", path=final)
    except Exception as e:
        j.update(status="failed", log=[str(e)[:300]])
    finally: shutil.rmtree(d, ignore_errors=True)

@app.get("/")
def root(): return {"service": "lipsync-wav2lip", "ok": True}
@app.get("/health")
def health(): return {"ok": True}
@app.post("/lipsync")
def submit(req: dict, x_api_key: str = Header(None)):
    auth(x_api_key); jid = uuid.uuid4().hex; JOBS[jid] = {"status": "pending"}
    threading.Thread(target=work, args=(jid, req), daemon=True).start(); return {"job_id": jid}
@app.get("/status/{jid}")
def status(jid: str, x_api_key: str = Header(None)):
    auth(x_api_key); j = JOBS.get(jid)
    if not j: return {"status": "not_found"}
    out = {k: v for k, v in j.items() if k != "path"}
    if out.get("status") == "running": out["status"] = "pending"      # n8n only needs pending / done / failed
    return out
@app.get("/video/{jid}")
def video(jid: str, x_api_key: str = Header(None)):
    auth(x_api_key); j = JOBS.get(jid)
    if not j or j.get("status") != "done": raise HTTPException(404, "not ready")
    return FileResponse(j["path"], media_type="video/mp4")
