# Voices

## What a voice is

A voice is a small float32 array of speaker latents:

```
voice_emb : (1, n_voice_queries, d_model)     # e.g. (1, 10, 768) — ~30 KB
```

That array is the **entire** speaker conditioning. At generation time it is
prepended to the sequence and nothing else about the speaker is involved: no
reference transcript, no in-context audio prompt, no teacher-forced frames. Two
runs with the same latents and the same seed produce the same speaker.

With no voice at all, the model falls back to a learned *unconditional* prefix
(`null_voice_emb.npy`). That is a valid mode — it produces natural speech — but
the identity is whatever the model picks and is not stable across runs.

## Built-in voices

Eight presets ship with the weights repo, each tagged by gender, age and
register/tone so you can pick one by ear or by filter. `maichi` (Mai Chi) is
the default used throughout this README and the demos.

| id | name | gender | tags |
|---|---|---|---|
| `maichi` | Mai Chi | nữ | trẻ · kể chuyện · nhẹ nhàng · thân thiện |
| `baotrang` | Bảo Trang | nữ | trưởng thành · tin tức · rõ ràng · trung tính |
| `kimoanh` | Kim Oanh | nữ | trung niên · kể chuyện · ấm áp · truyền cảm |
| `hamy` | Hà My | nữ | trẻ · hoạt hình · cao · biểu cảm |
| `giahuy` | Gia Huy | nam | trẻ · kể chuyện · trầm ấm · tâm tình |
| `huuduc` | Hữu Đức | nam | lớn tuổi · kể chuyện · trầm · điềm đạm |
| `quangminh` | Quang Minh | nam | trẻ · tin tức · rõ ràng · dứt khoát |
| `tiendat` | Tiến Đạt | nam | trẻ · bình luận · sôi nổi · năng lượng cao |

`v.tags` (a `list[str]`) and `v.gender`/`v.display_name` read straight off each
pack's `meta.json` — see [Voice pack format](#voice-pack-format) below.

## Using voices

```python
from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")

tts.list_voices()                    # ['maichi', 'baotrang', ...]

v = tts.load_voice("maichi")
v.emb.shape                          # (1, 10, 768)
v.language, v.description, v.preview_path

tts.synthesize("Xin chào.", voice="maichi")   # by name
tts.synthesize("Xin chào.", voice=v)        # by object
tts.synthesize("Xin chào.", voice=v.emb)    # by raw array
tts.synthesize("Xin chào.", voice=None)     # unconditional
```

`cfg_scale > 1` sharpens the identity by guiding away from the unconditional
branch, at twice the per-frame cost:

```python
tts.synthesize("Xin chào.", voice="maichi", cfg_scale=2.0)
```

## Cloning your own voice

**Not available in this release.** The latents above are produced by a voice
encoder that reads a reference clip, and that encoder is not published. This
package can load voices; it cannot create them from audio. There is no flag,
environment variable, or optional dependency that enables it — the code path does
not exist in the package, and the graph is not in the weights repo.

To get latents for your own speaker, clone a voice at
**[platform.zeroweight.ai/audio](https://platform.zeroweight.ai/audio)**, open
the voice library, and press the download button on the voice's row. You get a
zip holding one folder:

```
my-speaker/
  voice.npz      # the latents — this is the voice
  voice.bin      # the same latents as raw float32, for the browser demo
  preview.wav    # a sample of the voice
  meta.json      # display name, language, tags
  README.txt     # these instructions, in the folder
```

**Do not unzip it.** Every surface below takes the file as it came down, copies
what it needs out of it, and keeps the voice — so this is a one-time step, not
something to repeat on every run.

### Gradio demo

Open the voice picker's second tab, **Nhập giọng của bạn**, and drop the `.zip`
on it. The voice is installed, selected, and in the picker from then on. To
install one at startup instead:

```bash
python webui/app.py --voice ~/Downloads/my-speaker.zip     # repeatable
```

### Browser demo

Same place — the voice card's **Nhập giọng của bạn** tab — and the same gesture:
drop the `.zip` on it, or click to pick it. The file is read in the page;
nothing is uploaded. The voice is kept in IndexedDB, so it is in the picker on
the next visit too, and each one has an × to remove it again.

### Python

```python
from zerotts import ZeroTTS

tts = ZeroTTS.from_pretrained("zeroweight-ai/ZeroTTS")
tts.add_voices("~/Downloads/my-speaker.zip")   # once, ever
tts.list_voices()                              # ['baotrang', ..., 'my-speaker']
tts.synthesize("Xin chào.", voice="my-speaker")
```

`add_voices` unpacks the zip into `~/.zerotts/voices` (override with
`ZEROTTS_VOICES_HOME`) and every later `ZeroTTS` finds it there without being
told — which is why the call is not needed again. It also takes an
already-unpacked pack directory, or a directory of them, and uses those where
they lie rather than copying.

Each pack is validated against these weights as it is added: one built for a
different model raises here, and is not left installed, rather than generating a
confident stranger later. Added names shadow bundled ones, so a voice called
`maichi` means yours.

`zerotts.install_voice_zip(path, dest)` is the unpacking on its own, if you want
to install without a model in memory; `zerotts.installed_voices()` lists what is
already there.

### By hand

A pack dropped into `voices/<name>/` inside a local model directory is found by
the ordinary loader, with no API call at all. Or skip packs entirely and pass
the array: `tts.synthesize("...", voice=emb)`.

## Voice pack format

```
voices/
  index.json                     # manifest of every voice
  <name>/
    voice.npz                    # required
    voice.bin                    # optional: raw f32, for the browser demo
    preview.wav                  # optional
    meta.json                    # optional
```

`voice.npz` holds exactly two arrays:

| key | dtype | shape |
|---|---|---|
| `n_voice_queries` | int64 | scalar |
| `voice_emb` | float32 | `(1, n_voice_queries, d_model)` |

`voice.bin` is the same `voice_emb` as raw little-endian float32 in C order. It
exists so the browser demo can `fetch` + `new Float32Array` without a zip
parser; the Python loader ignores it and `from_pretrained` does not download it.
A pack with only a `voice.bin` and no `voice.npz` is a browser-demo voice, not a
voice pack — `list_voices` keys off `voice.npz`.