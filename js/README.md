# ZeroTTS browser demo

Runs ZeroTTS client-side with [`onnxruntime-web`](https://onnxruntime.ai/). No
server, no upload — weights are fetched from the Hugging Face CDN and everything
executes on the viewer's machine.

This is a **demo in this repository**, not a published npm package.

```bash
npm install
npm run dev        # http://localhost:5173
npm run typecheck
npm run build      # static bundle in dist/
```

> **Verified against the real model.** Driven through `onnxruntime-node`, this
> port produces frame codes **bit-identical** to the Python package for the same
> text, voice and random draws (`test/frames.mjs`). It has not yet been run in an
> actual browser, where the differences are the ORT build and the audio path
> rather than the model logic.

## The download

**~820 MB** on the default (ggml/GGUF) backend — a 772 MB fp32 GGUF plus the
45 MB ONNX codec decoder — fetched once and persisted via the Cache API. The
onnxruntime backend fetches ~900 MB of fp32 graphs instead. Smaller quantized
GGUFs are selectable (q8_0 at 206 MB, q4_0 at 124 MB) but are *slower*, not
faster — see below. The UI states the size before downloading either way. See
[../docs/BROWSER.md](../docs/BROWSER.md).

## Backends

The demo ships two, and **ggml/GGUF is the default**:

| | download | realtime | notes |
|---|---:|---:|---|
| **ggml/GGUF fp32** (default) | 772 MB | 6.1x | bit-exact against ONNX; no CFG |
| ggml/GGUF q8_0 | 206 MB | 3.6x | 4.9% of codes drawn differently |
| ggml/GGUF q4_0 | 124 MB | 4.1x | 36.5% differently |
| onnxruntime-web | ~858 MB | 4.2x | the only one with CFG |

Realtime figures at 4 threads on one machine; the ggml backend takes
`min(hardwareConcurrency, 8)` and gains from the extra threads, onnxruntime
caps at 4 and does not.

`src/ggmlBackend.ts` is the browser-side wrapper for [`../cpp/`](../cpp/) — same
`generateFrames` signature as `synthesizer.ts`, verified to produce identical
frame codes at fp32. Everything downstream of frame generation (chunking, the
streaming codec session, inter-segment silence) is shared, so the two backends
differ only in `FrameSource`. The codec itself stays on onnxruntime either way.

`bench-ggml.html` runs both side by side. Two things `../cpp/README.md`
documents that are worth knowing here: quantized weights are *slower* than fp32
in WebAssembly — which is why the default is the *unquantized* GGUF, and the
cross-attention K/V recompute that used to dominate `prefix_step.onnx` has been
fixed in the ONNX graphs too — worth ~1.28x at the segment lengths `chunkText`
produces.

## Layout

| File | Role |
|---|---|
| `src/synthesizer.ts` | the two-calls-per-frame loop — see [../docs/RUNTIME.md](../docs/RUNTIME.md) |
| `src/chunking.ts` | long-form segmentation, port of `zerotts.chunking` |
| `src/textNorm.ts` | Vietnamese text normalization, port of `zerotts.text_norm` |
| `src/codec.ts` | MOSS decoder: batch + KV-cached streaming |
| `src/tokenizer.ts` | BPE over `tokenizer.json` |
| `src/loader.ts` | create sessions from the downloaded graphs |
| `src/repo.ts` | resolve repo URLs, size the download, load voices |
| `src/worker.ts` | the Web Worker the model runs in |
| `src/workerClient.ts` | main-thread handle on that worker |
| `src/cache.ts` | download progress + Cache API persistence |
| `src/player.ts` | AudioWorklet ring buffer, WAV export |
| `src/rng.ts` | seedable PRNG — the sampler's draws are graph *inputs* |
| `src/samples.ts` | the sample texts, shared with the Python UI |
| `src/voicePack.ts` | read voice packs out of a dropped `.zip` |
| `src/voiceStore.ts` | keep imported voices in IndexedDB across visits |
| `src/main.ts` | demo UI wiring (imports no runtime code) |
| `src/ggmlBackend.ts` | the ggml/GGUF backend (see [../cpp/](../cpp/)) |
| `src/benchGgml.ts` | the ONNX-vs-ggml A/B page |

The model runs in a Web Worker: ORT-web's WASM backend computes on the calling
thread, and two graph calls per 80 ms frame on the UI thread freeze the tab for
the whole take. `main.ts` therefore imports nothing that pulls in
`onnxruntime-web` — it sends text to the worker and gets audio chunks back. See
[../docs/BROWSER.md](../docs/BROWSER.md) for the two subtleties (cancellation
needs a macrotask; chunks are transferred, not copied).

## Live playback

**Phát ngay trong lúc đang tạo** is on by default: chunks go to an AudioWorklet
ring buffer as they are decoded, so the take is audible while it is still being
made. Turn it off on a machine that generates slower than realtime — the buffer
runs dry between chunks and the worklet plays silence into the gaps, which
sounds like a broken model rather than a slow one. Off, the chunks are still
collected and the WAV still assembled; they are simply not pushed at the
speakers. The choice is remembered in `localStorage`.

(`webui/app.py` has the same checkbox, and skips opening its stream entirely
when it is off.)

## Your own voices

The voice card's second tab, **Nhập giọng của bạn**, takes the `.zip` a
[platform.zeroweight.ai](https://platform.zeroweight.ai/audio) voice downloads
as. Drop it on the page — no unzipping — and the voice joins the picker under
its own group.

Nothing is uploaded. `src/voicePack.ts` reads the archive in the page: the pack
folders inside it, then `voice.bin` when it is there (the raw float32 array) or
`voice.npz` when it is not, which means unzipping a second time and parsing
`voice_emb.npy`. Both zip layers can be stored or deflated, the latter via
`DecompressionStream`. `__MACOSX/` and other file-manager droppings are skipped.
The latents then go to the worker in the `generate` message, as `voiceEmb`
instead of a `voiceName` it would fetch.

`src/voiceStore.ts` keeps the imported voices in IndexedDB — the browser's
answer to the Python side copying a pack into `~/.zerotts/voices` — so they are
in the picker on the next visit, and each one has an × to remove it. Storage
being unavailable (a private window, a full quota) costs persistence and nothing
else.

A pack whose latents are the wrong length for the loaded weights is dropped with
a message rather than offered — wrong latents are the right dtype and rank, so
they would generate cleanly in a voice that is nobody's. See
[../docs/VOICES.md](../docs/VOICES.md).

## Parity checks

Two, both of which have caught real bugs:

**Frame codes vs. Python** — the end-to-end one. Sampling happens inside
`local_frame_decode.onnx` and takes its random draws as graph *inputs*, so the
caller owns the randomness and an exact cross-language comparison is possible:

```bash
npm install --no-save onnxruntime-node
node --experimental-strip-types test/frames.mjs /path/to/model
python test/py_frames.py /path/to/model
```

Frame codes must match element for element. (Audio *samples* may differ in the
last bits at chunk seams when comparing streaming to batch decode — that is the
KV-cached decoder, not an error — but the codes must not.)

**Text normalization vs. Python** — see below.

**ggml vs. ONNX** — the third, and the one that keeps the two backends honest
about each other. An fp32 GGUF must reproduce the ONNX frame codes exactly:

```bash
npm install --no-save onnxruntime-node
npm run parity:ggml
```

## Normalizer parity

`src/textNorm.ts` and `src/zerotts/text_norm/vi_normalizer.py` are
hand-maintained copies of the same rules. Nothing but a test keeps them in
step, and a divergence is invisible in normal use — it shows up only as the
browser and the package speaking the same sentence differently.

```bash
npm run parity        # 1262 cases, must be 100%
```

After changing either side, regenerate the corpus from Python and re-run:

```bash
python test/gen_cases.py > test/cases.json
npm run parity
```

CI runs it on every push.

## Cross-origin isolation

`vite.config.ts` sets COOP/COEP headers so `SharedArrayBuffer` is available and
onnxruntime-web can use multi-threaded WASM. Without them it silently falls back
to a single thread and generation is several times slower. **Whatever hosts the
built bundle must send the same two headers** — GitHub Pages does not, so a Pages
deployment will be slow unless you add a service-worker shim.

## Things that are easy to get wrong here

Three of these were real bugs in an earlier revision; they are called out
because each one fails *quietly*.

- **One codec session across all segments.** The backbone re-primes its
  `[voice | soa]` prefix per segment, but the codec's streaming decoder is
  causal and KV-cached — opening a fresh decoder per segment restarts that cache
  cold and clicks at every boundary. Each segment sounds fine in isolation.
- **The playback ring buffer must not lap itself.** Generation runs ~2x
  realtime, so the writer gains about a second of audio per second played. An
  unconditional modulo write silently overwrites unplayed samples once the lead
  exceeds the buffer. It is sized for the model's 120 s ceiling and reports an
  overflow rather than wrapping. A read/write index pair also makes "full" look
  identical to "empty"; monotonic counters are used instead.
- **int64 must literally be a `BigInt64Array`.** ORT rejects anything else
  ("A int64 tensor's data must be type of function BigInt64Array()"). A graph's
  int64 *output* is not guaranteed to come back as one across ORT-web versions
  and execution providers, and `outputs.x.data as BigInt64Array` is a cast that
  asserts rather than checks — so a wrong type survives until that value is fed
  back in as an input, which surfaces during `warmup()` at load time. Every int64
  input goes through `toBigInt64` / `i64` for that reason.
- **One RNG across segments.** Sampling draws are graph *inputs*, so a fresh
  `Rng(seed)` per segment replays the identical draw sequence for every segment.

## Known gaps

- WASM only — there is no WebGPU path. Its kernels are not bit-identical to the
  CPU path, and since this model samples *inside* the graph, small numeric
  differences change which token is drawn, which showed up as degraded output
  rather than as an error.
