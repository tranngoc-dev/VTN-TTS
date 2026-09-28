"""Gradio demo for ZeroTTS.

    pip install "zerotts[webui]"
    python webui/app.py
    python webui/app.py --model ./local_model_dir
    python webui/app.py --voice ~/Downloads/my-voice.zip

The page is deliberately split in two. The BASIC view is text → voice → player
and nothing else, so someone who just wants to hear a sentence never meets a
sampler knob. Everything technical — sampling parameters, the run log, the
segments actually sent to the model — lives under "Tuỳ chọn nâng cao".

Voice selection is a picker over the precomputed voice packs shipped with the
weights, plus any the user has installed. There is no "upload a reference clip"
control, because there is nothing behind it — the voice encoder is not part of
this release. What the picker's second tab takes is the finished article: the
.zip a zeroweight.ai voice downloads as, which it unpacks into the install
directory so it is simply there next time. See docs/VOICES.md.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import os
import sys

import gradio as gr

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _HERE)

# Run from a checkout, use the checkout's zerotts.
#
# This file only ever ships inside the repository, so `python webui/app.py` is
# always someone standing in a clone — and if they also have the published
# package installed, site-packages would win and they would be running a
# different version of the library than the one they can see. That failed
# loudly the first time the UI used a method the release did not have yet, and
# silently every time before that.
_SRC = os.path.join(_ROOT, "src")
if os.path.isfile(os.path.join(_SRC, "zerotts", "__init__.py")):
    sys.path.insert(0, _SRC)

import audio_stream  # noqa: E402
import engine  # noqa: E402

BANNER_PATH = os.path.join(_ROOT, "docs", "assets", "banner.png")

DEFAULT_TEXT = "Xin chào tất cả mọi người. Giọng nói này được tạo ra bởi ZeroTTS."

# The panel is the only place this is explained, so it carries the whole
# recipe: where a pack comes from, what the folder has to look like, and the
# two shapes of path that work. Someone pasting a path here has a folder open
# next to the browser and no reason to have read docs/VOICES.md.
USER_VOICE_HELP = (
    "**1.** Nhân bản giọng ở [zeroweight.ai](https://zeroweight.ai), mở "
    "**thư viện giọng**, bấm nút tải về ở dòng giọng đó.\n\n"
    "**2.** Kéo thả file `.zip` vừa tải vào ô dưới đây. "
    "Không cần giải nén — mọi thứ còn lại tự động.\n\n"
    "Giọng được lưu vào máy, nên lần sau mở lại là đã có sẵn trong danh sách."
)

# Why anyone would turn live playback off. The honest reason is that the
# transport cannot help: a machine that generates slower than realtime drains
# the browser's buffer faster than we fill it, and the player stalls and resumes
# mid-sentence. The finished file is unaffected, and that is the point worth
# making — turning this off costs nothing but the preview.
LIVE_PLAYBACK_NOTE = (
    "Nếu máy chậm (tạo chậm hơn thời gian thực), hãy **tắt** tuỳ chọn này — "
    "phát trực tiếp sẽ bị ngắt quãng giữa chừng. Bản hoàn chỉnh bên dưới "
    "không bị ảnh hưởng."
)

MODE_VOICE = "voice"
MODE_UNCOND = "uncond"
MODE_CHOICES = [
    ("Dùng gói giọng đọc (Voice pack)", MODE_VOICE),
    ("Tự do (Không dùng giọng đọc)", MODE_UNCOND),
]

_mounted_apps: set = set()


# ── chrome ───────────────────────────────────────────────────────────────────

def banner_html() -> str:
    """The banner, inlined as a data URI so it needs no static route and no
    `allowed_paths` entry. Degrades to a wordmark if the file is missing."""
    try:
        with open(BANNER_PATH, "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return "<div class='zt-banner zt-banner-text'><span>Zero</span>TTS</div>"
    return f"<div class='zt-banner'><img src='{src}' alt='ZeroTTS' /></div>"


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
)

# One brand ramp (the banner's orange) plus a violet accent, then a handful of
# structural overrides. The two list-shaped panels (history, templates) are
# gr.Dataset with two hidden Textbox columns, which Gradio renders as a <table>
# of <tr class="tr-body"> — the .zt-list rules below turn each of those rows
# into a two-line card: title cell on top, preview cell under it. Two columns
# and not one because the Dataset frontend cuts every CELL at 60 characters
# (Example-*.js `slice(0, 60) + "..."`), so a title and a readable preview do
# not fit in a single one.
CSS = """
:root, .gradio-container {
  --zt-brand: #f4530c;      /* the banner's orange */
  --zt-brand-2: #ff8a3d;
  --zt-soft: rgba(244, 83, 12, .09);
}
/* The wash goes full-bleed, not on the (centred, max-width) container where it
   would stop dead at the edges and read as a stray rectangle. <gradio-app> is
   the element that paints the page colour — body sits behind it. */
gradio-app {
  background-image:
    radial-gradient(1200px 560px at 4% -8%, rgba(244,83,12,.20), transparent 62%),
    radial-gradient(1000px 500px at 99% 0%, rgba(124,58,237,.18), transparent 64%) !important;
  background-attachment: fixed !important;
  background-repeat: no-repeat !important;
}
/* <gradio-app> lays its child out with flex and no justify-content, so the
   capped-width container sticks to the left edge until it is given auto
   margins — max-width alone does not centre it. */
.gradio-container {
  max-width: 1180px !important; margin: 0 auto !important;
  background: transparent !important;
}
footer { display: none !important; }

/* banner */
.zt-banner { margin: 0 auto .25rem; text-align: center; }
.zt-banner img {
  width: 100%; max-width: 560px; height: auto; display: block; margin: 0 auto;
  border-radius: 18px;
}
.zt-banner-text { font-size: 2.4rem; font-weight: 800; letter-spacing: -.02em; }
.zt-banner-text span { color: var(--zt-brand); }
.zt-headline { text-align: center; margin: .9rem auto 1.1rem !important; max-width: 62ch; }
.zt-headline h1 {
  font-size: 1.3rem !important; font-weight: 700 !important;
  line-height: 1.35 !important; letter-spacing: -.01em;
  margin: 0 !important; padding: 0 !important;
}

/* cards */
.zt-card {
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 20px !important;
  padding: 1.15rem 1.15rem 1.25rem !important;
  background: var(--background-fill-primary) !important;
  box-shadow: 0 10px 30px -22px rgba(20, 20, 40, .55);
}
/* Section titles read as headings, not as body copy: heavier, slightly larger,
   and led by a short brand bar so the eye finds the start of each card. */
.zt-card-title {
  display: flex; align-items: center; gap: .55rem;
  margin: 0 0 .35rem !important;
}
.zt-card-title p, .zt-card-title h1, .zt-card-title h2, .zt-card-title h3 {
  font-weight: 800 !important; font-size: 1.14rem !important;
  letter-spacing: -.015em; line-height: 1.3 !important;
  margin: 0 !important; padding: 0 !important;
}
.zt-card-title::before {
  content: ""; flex: none; width: .28rem; height: 1.15rem; border-radius: 999px;
  background: linear-gradient(180deg, var(--zt-brand-2), var(--zt-brand));
}
.zt-hint {
  color: var(--body-text-color-subdued); font-size: .84rem;
  margin: 0 0 .55rem !important;
}

/* the primary action */
#zt-generate {
  background: linear-gradient(135deg, var(--zt-brand-2), var(--zt-brand)) !important;
  border: none !important; color: #fff !important;
  font-weight: 700 !important; font-size: 1.02rem !important;
  padding: .8rem 1rem !important;
  box-shadow: 0 12px 24px -14px rgba(244, 83, 12, .95);
}
#zt-generate:hover { filter: brightness(1.06); }
.zt-actions { margin-top: .35rem; align-items: stretch; }
.zt-actions button { min-height: 3rem; }

/* the run-detail accordion nested inside the text card: a quiet disclosure,
   not a second panel competing with the card it sits in */
.zt-inline-accordion {
  border: none !important; background: transparent !important;
  margin-top: .5rem !important; padding: 0 !important;
}
.zt-inline-accordion > .label-wrap {
  padding: 0 !important; font-size: .85rem;
  color: var(--body-text-color-subdued) !important;
}
/* the block's own status wrap keeps its `hide` class even when the accordion is
   open; with the accordion's padding gone it shows through as a stray scrollbar */
.zt-inline-accordion > .wrap.hide { display: none !important; }

/* list-shaped datasets (history, templates) */
.zt-list .table-wrap { border: none !important; overflow: visible !important; }
.zt-list table { width: 100% !important; border-collapse: separate; }
.zt-list thead, .zt-list .tr-head { display: none !important; }
.zt-list tbody { display: flex; flex-direction: column; gap: .5rem; }
.zt-list .tr-body {
  display: flex !important; flex-direction: column; align-items: stretch;
  flex: none;  /* rows are flex items now; without this they squash */
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 14px !important; background: var(--background-fill-secondary) !important;
  transition: transform .12s ease, border-color .12s ease, background .12s ease;
}
.zt-list .tr-body:hover {
  border-color: var(--zt-brand) !important; background: var(--zt-soft) !important;
  transform: translateY(-1px);
}
.zt-list td {
  border: none !important; text-align: left !important;
  padding: .1rem .85rem !important; max-width: none !important;
  white-space: pre-wrap;  /* the separators in a row title are real spaces */
}
.zt-list td:first-child {
  padding-top: .6rem !important;
  font-weight: 700; font-size: .93rem; color: var(--body-text-color);
}
.zt-list td:last-child {
  padding-bottom: .6rem !important;
  font-size: .84rem; color: var(--body-text-color-subdued);
}
.zt-list td > * { text-align: left !important; }
.zt-list .paginate { justify-content: flex-start; font-size: .8rem; }
#zt-history tbody { max-height: 27rem; overflow-y: auto; overflow-x: hidden; }
#zt-templates tbody { flex-direction: row; flex-wrap: wrap; }
#zt-templates .tr-body { width: calc(50% - .25rem); }
@media (max-width: 820px) { #zt-templates .tr-body { width: 100%; } }

/* players */
.zt-player audio { width: 100%; }
.zt-live { margin-top: .2rem; }
/* The live element is a plain <audio> with the browser's own controls (see
   audio_stream.py) — color-scheme is the only way to stop Chrome painting it
   bright white in the middle of the dark theme. */
.zt-live audio { width: 100%; }
.dark .zt-live audio { color-scheme: dark; }
.zt-foot {
  text-align: center; font-size: .84rem; line-height: 1.5;
  color: var(--body-text-color-subdued);
  margin: .9rem 0 0 !important;
}
"""


# Gradio 6 moved `theme`/`css` off the Blocks constructor and onto
# launch()/mount_gradio_app(); 4.x and 5.x only accept them on the constructor.
# pyproject supports gradio>=4.44, so decide per install rather than pinning.
_STYLE = {"theme": THEME, "css": CSS}
_STYLE_ON_MOUNT = "css" in inspect.signature(gr.mount_gradio_app).parameters
_STYLE_ON_BLOCKS = {} if _STYLE_ON_MOUNT else _STYLE


def _ensure_stream_route(app) -> None:
    """Register audio_stream's route on `app` once. Idempotent, and a no-op
    before a server exists."""
    if app is None or id(app) in _mounted_apps:
        return
    _mounted_apps.add(id(app))
    audio_stream.mount(app)


# ── voices ───────────────────────────────────────────────────────────────────

def refresh_voices():
    """Dropdown choices. Prefers `maichi` as the default so a fresh page load
    always starts on the same voice the README and samples use."""
    choices = engine.voice_choices(None)
    names = [v for _, v in choices]
    default = "maichi" if "maichi" in names else (names[0] if names else None)
    return gr.update(choices=choices, value=default)


def load_user_voices(path, current):
    """The "your voices" panel: install a dropped zip, then move the picker onto
    the first voice it brought in.

    Selecting it rather than only refreshing the list is the point of the
    panel — someone who has just dropped a file in wants to hear that voice, not
    to go and find it in a dropdown that grew by one.

    The upload component is cleared either way (the last output): leaving the
    file sitting in the box after a successful install reads as "not done yet",
    and after a failed one it invites pressing the same thing again.
    """
    names, message = engine.add_voices(path)
    if not names:
        return gr.update(), gr.update(), gr.update(), message, None
    choices = engine.voice_choices(None)
    chosen = names[0] if names[0] in [v for _, v in choices] else current
    preview, meta = on_voice_change(chosen)
    return gr.update(choices=choices, value=chosen), preview, meta, message, None


def on_voice_change(name):
    """Preview + description for the selected voice.

    Returns a path or None — never a missing file, which is what makes
    gr.Audio render an error box instead of an empty player.
    """
    return engine.voice_preview_path(name), engine.voice_info(name)


# ── templates ────────────────────────────────────────────────────────────────

# Gradio's Dataset frontend cuts every cell at 60 characters and appends "...",
# so shorten here instead and use a nicer ellipsis.
CELL_CHARS = 57


def _shorten(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= CELL_CHARS else text[:CELL_CHARS - 1].rstrip() + "…"




# ── history ──────────────────────────────────────────────────────────────────

def _history_rows(paths: list) -> list:
    """One card per saved file: voice + when + length, then the text that was
    spoken. Clicking a card loads it into the main player."""
    rows = []
    for path in paths:
        meta = engine.generated_meta(path)
        head = f"🔊  {engine.voice_display_name(meta['voice']) or meta['voice']}"
        head += f"  ·  {meta['when']}"
        if meta["seconds"]:
            head += f"  ·  {meta['seconds']:.0f}s"
        rows.append([_shorten(head), _shorten(meta["text"]) or "—"])
    return rows


def refresh_history():
    files = engine.list_generated()
    return gr.update(samples=_history_rows(files)), files


def play_selected(evt: gr.SelectData, file_list):
    """History click → the main output player, so there is exactly one place
    audio comes from on this page."""
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if file_list and 0 <= idx < len(file_list):
        return file_list[idx]
    return None


# ── generation ───────────────────────────────────────────────────────────────

def clear_players():
    """Blank both players as their own event (not part of the generate run) so a
    new generation can't be heard on top of the previous one's buffered audio."""
    return audio_stream.player_html(None), None


def generate_ui(text, voice_name, mode, max_chunk_sec, cfg_scale, temperature,
                topk, topp, repetition_penalty, eoa_extra_frames, speak_live,
                file_list):
    """Streams audio out through webui/audio_stream.py — a single continuous WAV
    response — NOT through gr.Audio(streaming=True).

    That component is not a continuous waveform: Gradio turns every yielded chunk
    into its own HLS segment with an independent AAC encode, so each one carries
    encoder priming at the front and zero padding at the back. Concatenated, every
    chunk boundary clicks; and our first chunks are 1-4 codec frames (0.08-0.32 s),
    shorter than AAC's own priming, which is why the first chunk appears to repeat.
    The saved .wav is always fine — it is the transport that is broken.

    ``speak_live`` off skips the stream entirely rather than opening one nobody
    listens to: on a machine slower than realtime the player stalls mid-sentence,
    and the generation itself is unchanged either way — the same chunks are
    produced, the same file is saved, they simply are not pushed anywhere.

    Outputs: (live player HTML, completed-file player, status, history dataset,
    history state, segments box).
    """
    use_voice = mode == MODE_VOICE
    if use_voice and not voice_name:
        yield (gr.update(), gr.update(), "Hãy chọn một giọng đọc trước.",
               gr.update(), file_list, gr.update())
        return
    if len(text or "") > engine.MAX_TEXT_CHARS:
        yield (gr.update(), gr.update(),
               f"Văn bản quá dài ({len(text)} ký tự, tối đa {engine.MAX_TEXT_CHARS}).",
               gr.update(), file_list, gr.update())
        return

    segments = engine.get_text_segments(text, max_chunk_sec=max_chunk_sec)
    segments_text = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(segments))
    yield gr.update(), gr.update(), "Đang tạo…", gr.update(), file_list, segments_text

    sample_rate = engine.get_sample_rate()
    sid = audio_stream.open_stream(sample_rate) if speak_live else None
    if sid:
        # Show the player before the first chunk exists: the route blocks until
        # audio arrives, so the browser connects and starts buffering right away.
        yield (audio_stream.player_html(sid), gr.update(), "Đang tạo…",
               gr.update(), file_list, gr.update())

    result: dict = {}
    n_samples = 0
    try:
        try:
            for _sr, chunk in engine.generate_stream(
                text=text, voice_name=voice_name, max_chunk_sec=max_chunk_sec,
                cfg_scale=float(cfg_scale), audio_temperature=temperature,
                audio_topk=int(topk), audio_topp=topp,
                audio_repetition_penalty=repetition_penalty,
                eoa_extra_frames=int(eoa_extra_frames), use_voice=use_voice,
                result=result,
            ):
                if sid:
                    audio_stream.push(sid, chunk)
                n_samples += chunk.shape[0]
                yield (gr.update(), gr.update(),
                       f"Đang tạo… {n_samples / sample_rate:.1f}s",
                       gr.update(), file_list, gr.update())
        except Exception as exc:
            yield gr.update(), gr.update(), f"Lỗi: {exc}", gr.update(), file_list, gr.update()
            return
    finally:
        # Ends the HTTP response cleanly, including when the run is cancelled by
        # the Stop button (the generator is closed, which lands us here).
        if sid:
            audio_stream.close(sid)

    files = engine.list_generated()
    saved = result.get("path")
    yield (
        gr.update(),
        saved,
        f"Xong — {n_samples / sample_rate:.1f}s. Đã lưu vào {saved}" if saved else "Xong.",
        gr.update(samples=_history_rows(files)),
        files,
        gr.update(),
    )


with gr.Blocks(title="ZeroTTS", **_STYLE_ON_BLOCKS) as demo:
    gr.HTML(banner_html())
    gr.Markdown(
        "# ZeroTTS - Chuyển văn bản tiếng Việt thành giọng nói tự nhiên, "
        "nhanh và realtime trên CPU",
        elem_classes="zt-headline",
    )

    history_state = gr.State([])

    with gr.Row():
        # ── basic view: text → player, and nothing else ──────────────────────
        with gr.Column(scale=3):
            with gr.Column(elem_classes="zt-card"):
                gr.Markdown("Nhập văn bản", elem_classes="zt-card-title")
                gr.Markdown(
                    f"Tối đa {engine.MAX_TEXT_CHARS} ký tự.",
                    elem_classes="zt-hint",
                )
                text_box = gr.Textbox(
                    value=DEFAULT_TEXT, label=None, show_label=False,
                    lines=7, max_lines=20, container=False,
                    placeholder="Nhập văn bản tiếng Việt…",
                )
                with gr.Row(elem_classes="zt-actions"):
                    generate_btn = gr.Button("🎙️  Tạo giọng nói", variant="primary",
                                             elem_id="zt-generate", scale=3)
                    stop_btn = gr.Button("Dừng", scale=1)

                # What the run did, next to the run itself rather than down in
                # the global settings accordion — it is about this text, not
                # about how the model is configured.
                with gr.Accordion("Xem chi tiết lần chạy", open=False,
                                  elem_classes="zt-inline-accordion"):
                    gen_status = gr.Textbox(label="Trạng thái", interactive=False)
                    segments_box = gr.Textbox(
                        label="Các đoạn được gửi tới mô hình "
                              "(sau khi tách câu và làm sạch)",
                        interactive=False, lines=6,
                    )

            with gr.Column(elem_classes="zt-card"):
                gr.Markdown("Nghe kết quả", elem_classes="zt-card-title")
                speak_live_checkbox = gr.Checkbox(
                    value=True, label="Phát ngay trong lúc đang tạo",
                    info=LIVE_PLAYBACK_NOTE, container=False,
                )
                # Hidden wholesale when the option is off, label and all — an
                # empty player sitting there through every run reads as broken
                # rather than switched off.
                with gr.Column(visible=True) as live_box:
                    # A plain <audio> fed by our own continuous-WAV route rather
                    # than gr.Audio(streaming=True) — see generate_ui's docstring
                    # and webui/audio_stream.py.
                    live_player = gr.HTML(value=audio_stream.player_html(None),
                                          elem_classes="zt-live")
                completed_audio = gr.Audio(
                    label="Bản hoàn chỉnh — tua và tải về được",
                    interactive=False, autoplay=False, elem_classes="zt-player",
                )
                gr.Markdown(
                    "**Nhân bản giọng nói (voice cloning)** không có trong bản mã "
                    "nguồn mở — bộ mã hoá giọng chưa được phát hành. Nhân bản "
                    "giọng ở [zeroweight.ai](https://zeroweight.ai), tải file "
                    "`.zip` về, rồi kéo thả vào tab **Nhập giọng của bạn** "
                    "bên phải.",
                    elem_classes="zt-foot",
                )

        # ── voice picker, then the takes it produced ─────────────────────────
        with gr.Column(scale=2):
            with gr.Column(elem_classes="zt-card"):
                gr.Markdown("Giọng đọc", elem_classes="zt-card-title")
                # Two tabs rather than two cards: choosing a voice and importing
                # one are the same job at different moments, and an imported
                # voice lands in the picker on the tab beside it.
                with gr.Tabs():
                    with gr.Tab("Chọn giọng"):
                        with gr.Row():
                            voice_dropdown = gr.Dropdown(
                                choices=[], label=None, show_label=False,
                                value=None, container=False, scale=5)
                            refresh_voices_btn = gr.Button("↻", scale=0, min_width=48)
                        voice_meta = gr.Markdown("", elem_classes="zt-hint")
                        voice_preview = gr.Audio(label="Nghe thử giọng",
                                                 interactive=False,
                                                 elem_classes="zt-player")

                    with gr.Tab("Nhập giọng của bạn"):
                        gr.Markdown(USER_VOICE_HELP, elem_classes="zt-hint")
                        # A file drop, not a path box: the user has the zip in a
                        # downloads folder and a browser in front of them, and
                        # asking them to find its absolute path is asking them
                        # to open a terminal. `type="filepath"` because the
                        # installer reads the archive off disk — Gradio has
                        # already saved it there by the time this runs.
                        user_voice_zip = gr.File(
                            label="Kéo thả file .zip giọng vào đây",
                            file_types=[".zip"], type="filepath", height=110,
                        )
                        user_voice_status = gr.Markdown("", elem_classes="zt-hint")

            with gr.Column(elem_classes="zt-card"):
                with gr.Row():
                    gr.Markdown("Đã tạo gần đây", elem_classes="zt-card-title")
                    refresh_history_btn = gr.Button("↻", scale=0, min_width=48)
                gr.Markdown("Bấm vào một mục để nghe lại ở trình phát chính.",
                            elem_classes="zt-hint")
                history_dataset = gr.Dataset(
                    components=[gr.Textbox(visible=False),
                                gr.Textbox(visible=False)], samples=[],
                    label=None, show_label=False, samples_per_page=20,
                    elem_id="zt-history", elem_classes="zt-list",
                )

    # ── advanced view: everything technical, closed by default ──────────────
    with gr.Accordion("Tuỳ chọn nâng cao", open=False):
        mode_radio = gr.Radio(
            choices=MODE_CHOICES, value=MODE_VOICE, label="Chế độ giọng đọc",
            info="Gói giọng đọc: Nhúng vector giọng chọn sẵn vào mô hình để giữ cố định giọng. "
                 "Không dùng giọng: Mô hình tự chọn giọng ngẫu nhiên giữa các đoạn "
                 "(dùng để kiểm thử chất lượng gốc).",
        )
        cfg_slider = gr.Slider(
            1.0, 4.0, value=1.0, step=0.1, label="Mức độ định hướng giọng (CFG Scale)",
            info="1.0 = Tắt. Giá trị trên 1.0 sẽ tăng độ trung thực của đặc trưng giọng đọc "
                 "nhưng tốn gấp 2 lần thời gian xử lý (chạy song song 2 luồng tính toán).",
        )
        with gr.Row():
            chunk_sec_slider = gr.Slider(
                5, 25, value=15, step=1,
                label="Độ dài tối đa mỗi đoạn câu (giây)",
                info="Tự động cắt bài văn dài thành các đoạn nhỏ. Giúp mô hình quản lý bộ nhớ đệm "
                     "hiệu quả và phát âm thanh trực tiếp (streaming) mượt mà.",
            )
            temperature_slider = gr.Slider(
                0.1, 1.5, value=0.8, step=0.05,
                label="Độ ngẫu nhiên âm thanh (Temperature)",
                info="Kiểm soát độ biến thiên giọng đọc. Giá trị thấp (<0.6) đọc chuẩn xác, "
                     "ít lỗi nhưng có thể máy móc; giá trị cao (>0.9) đọc tự nhiên, "
                     "nhiều biểu cảm hơn.",
            )
        with gr.Row():
            topk_slider = gr.Slider(
                1, 200, value=25, step=1, label="Tỷ lệ lấy mẫu Top-K",
                info="Chỉ giữ lại K âm thanh có xác suất cao nhất tại mỗi bước. Tắt các âm thanh "
                     "nhiễu rác. Giảm xuống 15-20 giúp bài đọc ổn định, tăng lên cho phép "
                     "nhiều biến thể.",
            )
            topp_slider = gr.Slider(
                0.1, 1.0, value=0.95, step=0.01, label="Tỷ lệ lấy mẫu Top-P",
                info="Giữ lại tập hợp âm thanh có tổng xác suất tích lũy đạt P (95%). "
                     "Phối hợp cùng Top-K để tự động mở rộng/thu hẹp tập lựa chọn "
                     "theo độ khó của từ.",
            )
        repetition_penalty_slider = gr.Slider(
            1.0, 2.0, value=1.2, step=0.05, label="Hình phạt lặp âm (Repetition penalty)",
            info="Trừ điểm xác suất với các mã âm thanh đã xuất hiện trong đoạn. "
                 "Ngăn chặn hiện tượng lặp lại từ hoặc nói vấp. Đặt 1.0 sẽ tắt hình phạt.",
        )
        eoa_extra_slider = gr.Slider(
            0, 4, value=1, step=1, label="Số khung âm giữ lại ở đuôi (Tail frames)",
            info="Số khung âm (mỗi khung 0.08 giây) được giữ lại sau tín hiệu dừng. "
                 "Giúp phần nhả hơi/kết thúc câu ở từ cuối cùng trọn vẹn và tự nhiên.",
        )
        gr.Markdown(
            "<sub>Dấu câu được chuẩn hóa trước khi tổng hợp: dấu `;` chuyển thành dấu phẩy `,`, "
            "dấu xuống dòng chuyển thành ngắt câu (trừ khi dòng đã kết thúc bằng dấu câu). "
            "Mô hình được huấn luyện trên dữ liệu lời nói chuẩn hóa, nên dòng xuống dòng "
            "không sửa đổi sẽ bị mất.</sub>"
        )

    refresh_voices_btn.click(fn=refresh_voices, outputs=[voice_dropdown]).then(
        fn=on_voice_change, inputs=[voice_dropdown],
        outputs=[voice_preview, voice_meta],
    )
    voice_dropdown.change(fn=on_voice_change, inputs=[voice_dropdown],
                          outputs=[voice_preview, voice_meta])

    gen_event = generate_btn.click(
        fn=clear_players, outputs=[live_player, completed_audio],
    ).then(
        fn=generate_ui,
        inputs=[text_box, voice_dropdown, mode_radio, chunk_sec_slider, cfg_slider,
                temperature_slider, topk_slider, topp_slider,
                repetition_penalty_slider, eoa_extra_slider, speak_live_checkbox,
                history_state],
        outputs=[live_player, completed_audio, gen_status, history_dataset,
                 history_state, segments_box],
    )
    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

    speak_live_checkbox.change(
        fn=lambda on: gr.update(visible=bool(on)),
        inputs=[speak_live_checkbox], outputs=[live_box],
    )

    user_voice_zip.upload(
        fn=load_user_voices, inputs=[user_voice_zip, voice_dropdown],
        outputs=[voice_dropdown, voice_preview, voice_meta, user_voice_status,
                 user_voice_zip],
    )

    refresh_history_btn.click(fn=refresh_history,
                              outputs=[history_dataset, history_state])
    history_dataset.select(fn=play_selected, inputs=[history_state],
                           outputs=[completed_audio])

    def _on_load():
        # Blocks is one shared graph across page loads, and this may have been
        # started by something other than __main__ (plain demo.launch(), `gradio
        # app.py`, an external mount) — the live-audio route has to exist on
        # whatever FastAPI app is actually serving us, or the player 404s.
        _ensure_stream_route(getattr(demo, "app", None))
        voice_update = refresh_voices()
        default = voice_update["value"]
        files = engine.list_generated()
        preview, meta = on_voice_change(default)
        return (
            voice_update,
            preview, meta,
            gr.update(samples=_history_rows(files)), files,
        )

    demo.load(
        fn=_on_load,
        outputs=[voice_dropdown, voice_preview, voice_meta,
                 history_dataset, history_state],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=engine.DEFAULT_MODEL)
    ap.add_argument(
        "--voice", action="append", default=[], metavar="PATH",
        dest="voices", help="A downloaded voice .zip, or a voice pack directory. "
             "Repeatable. The same thing the picker's import tab does, for a "
             "server that should start with them installed.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    engine.set_model(args.model)

    # Loading the packs also loads the model, which the first request would do
    # anyway — and doing it here means a bad --voice is reported at startup
    # rather than by an empty dropdown later.
    for path in args.voices:
        names, message = engine.add_voices(path)
        print(message)
        if not names:
            raise SystemExit(2)

    # Mounted onto our own FastAPI app rather than demo.launch(), so the live
    # audio route is registered before the server starts. Gradio goes at "/" —
    # audio_stream.STREAM_ROUTE assumes that.
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    _ensure_stream_route(app)
    app = gr.mount_gradio_app(app, demo.queue(), path="/",
                              **(_STYLE if _STYLE_ON_MOUNT else {}))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
