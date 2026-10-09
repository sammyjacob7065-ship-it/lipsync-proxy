# LipSync proxy: Wav2Lip (ONNX, CPU) for Render free (Docker web service)
Same calls the n8n lane uses: `POST /lipsync`, `GET /status/<id>`, `GET /video/<id>` (header `x-api-key`), `GET /health`.
Only the speaker's face changes, only inside that line's time window. Output is video only (assemble.py adds the voices).

## Deploy (same way as your video proxy)
1. Put `Dockerfile`, `requirements.txt`, `app.py` in a GitHub repo (new repo, e.g. `lipsync-proxy`).
2. Render > New > Web Service > that repo > Language **Docker** > Free.
3. Environment: `API_KEY` = a long random word.
4. When live, open `https://<name>.onrender.com/health` -> `{"ok":true}`.
5. Put the address and key in **LipSync: Build Jobs** (`lipSyncProxy`), or run `python set_lipsync_proxy.py <workflow.json> <url> <key>`.

## Expect (read this)
- First request downloads the model (about 145 MB, from Hugging Face) after every restart.
- Free Render is a small CPU (as far as I know about 0.1 CPU and 512 MB; not checked). If it is that small, one scene can take a long time or run out of memory. Test ONE short scene and watch the Render logs. n8n waits up to 60 minutes per scene (`lipSyncMaxWaitMin` in *LipSync: Judge Job*).
- It sleeps like your video proxy; *LipSync: Submit Job* retries for about 4-5 minutes while it wakes.
- If it is too slow or crashes: run the same app on your own PC (`uvicorn app:app --port 8000`, public with a free tunnel), or on a free Colab/Kaggle session. Nothing in n8n changes, only the address.

## Limits (Wav2Lip itself)
- Mouth area is 96x96 and soft; best on frontal faces.
- Two faces: the speaker's face is picked left to right using the order the characters are named in the scene's image text (`face_order`).
- Licence: Wav2Lip is personal / research / non-commercial; the ONNX weights state no licence. Check before monetising.
- `LIPSYNC_STUB=1` runs everything with a fake face and model (testing only).
