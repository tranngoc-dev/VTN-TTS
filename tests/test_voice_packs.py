"""Voice packs from outside the model directory.

A voice downloaded from the platform arrives as a **zip**, and the whole point
is that dropping that file in is the entire procedure — so what is tested here
is the shape of what people are actually handed, including the shapes a real
download and a real file manager produce. Nothing here needs weights.
"""

import json
import zipfile

import numpy as np
import pytest

from zerotts.voices import (
    discover_packs,
    install_voice_zip,
    installed_voices,
    is_voice_pack,
    load_voice_dir,
)

Q, D = 10, 768


def write_pack(directory, name, n_queries=Q, meta=None, preview=False):
    vdir = directory / name
    vdir.mkdir(parents=True)
    np.savez(vdir / "voice.npz",
             n_voice_queries=np.int64(n_queries),
             voice_emb=np.zeros((1, n_queries, D), dtype=np.float32))
    if meta is not None:
        (vdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if preview:
        (vdir / "preview.wav").write_bytes(b"RIFF")
    return vdir


def test_a_pack_directory_is_discovered_as_itself(tmp_path):
    vdir = write_pack(tmp_path, "my-speaker")
    assert is_voice_pack(vdir)
    assert discover_packs(vdir) == {"my-speaker": vdir}


def test_a_parent_directory_yields_every_pack_under_it(tmp_path):
    write_pack(tmp_path, "one")
    write_pack(tmp_path, "two")
    (tmp_path / "not-a-voice").mkdir()
    assert sorted(discover_packs(tmp_path)) == ["one", "two"]


def test_a_directory_with_no_packs_discovers_nothing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert discover_packs(tmp_path) == {}


def test_a_missing_directory_is_not_a_silent_empty_result(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover_packs(tmp_path / "nope")


def test_load_reads_metadata_and_preview(tmp_path):
    vdir = write_pack(tmp_path, "my-speaker", preview=True,
                      meta={"display_name": "My Speaker", "tags": ["nữ", "trẻ"],
                            "language": "vi"})
    voice = load_voice_dir(vdir, expect_queries=Q)
    assert voice.name == "my-speaker"
    assert voice.display_name == "My Speaker"
    assert voice.tags == ["nữ", "trẻ"]
    assert voice.emb.shape == (1, Q, D)
    assert voice.preview_path is not None


def test_load_works_without_the_optional_files(tmp_path):
    voice = load_voice_dir(write_pack(tmp_path, "bare"), expect_queries=Q)
    assert voice.display_name == "bare"
    assert voice.tags == []
    assert voice.preview_path is None


def test_a_pack_for_other_weights_is_refused(tmp_path):
    """Latents built against a different model have the right dtype and rank, so
    they would feed the graph cleanly and produce confident nonsense."""
    vdir = write_pack(tmp_path, "wrong", n_queries=Q + 1)
    with pytest.raises(ValueError, match="different weights"):
        load_voice_dir(vdir, expect_queries=Q)


def test_a_directory_without_a_npz_is_not_a_pack(tmp_path):
    (tmp_path / "bin-only").mkdir()
    (tmp_path / "bin-only" / "voice.bin").write_bytes(b"\0" * (Q * D * 4))
    assert not is_voice_pack(tmp_path / "bin-only")
    with pytest.raises(FileNotFoundError, match="not a voice pack"):
        load_voice_dir(tmp_path / "bin-only")


# ── the downloaded zip ───────────────────────────────────────────────────────

def zip_pack(tmp_path, name, extra=(), strip_top_level=False, **kwargs):
    """A zip shaped like the platform's download: `<name>/voice.npz` and
    friends, plus the README it ships with."""
    vdir = write_pack(tmp_path / "src", name, **kwargs)
    (vdir / "README.txt").write_text("how to use this voice", encoding="utf-8")
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(vdir.iterdir()):
            arcname = file.name if strip_top_level else f"{name}/{file.name}"
            archive.write(file, arcname)
        for junk in extra:
            archive.writestr(junk, b"junk")
    return path


def test_installing_a_zip_copies_the_pack_out_of_it(tmp_path):
    path = zip_pack(tmp_path, "my-speaker", preview=True,
                    meta={"display_name": "My Speaker"})
    installed = install_voice_zip(path, tmp_path / "home")
    assert list(installed) == ["my-speaker"]
    vdir = installed["my-speaker"]
    # The pack's files, and only those: the README the download ships with is
    # for the person, not for the loader.
    assert sorted(f.name for f in vdir.iterdir()) == [
        "meta.json", "preview.wav", "voice.npz"]
    assert load_voice_dir(vdir, expect_queries=Q).emb.shape == (1, Q, D)


def test_an_installed_voice_is_found_again_without_the_zip(tmp_path):
    """The install is the point: the file is dropped in once, not every run."""
    home = tmp_path / "home"
    install_voice_zip(zip_pack(tmp_path, "my-speaker"), home)
    assert list(installed_voices(home)) == ["my-speaker"]


def test_reinstalling_replaces_rather_than_duplicates(tmp_path):
    home = tmp_path / "home"
    path = zip_pack(tmp_path, "my-speaker")
    install_voice_zip(path, home)
    install_voice_zip(path, home)
    assert list(installed_voices(home)) == ["my-speaker"]


def test_a_zip_with_the_pack_at_its_root_is_named_after_the_file(tmp_path):
    path = zip_pack(tmp_path, "my-speaker", strip_top_level=True)
    assert list(install_voice_zip(path, tmp_path / "home")) == ["my-speaker"]


def test_finder_and_editor_droppings_are_not_installed(tmp_path):
    """A zip that has been round-tripped through macOS carries a parallel
    __MACOSX tree; following it would find every pack twice."""
    path = zip_pack(tmp_path, "my-speaker",
                    extra=("__MACOSX/my-speaker/._voice.npz", ".DS_Store"))
    installed = install_voice_zip(path, tmp_path / "home")
    assert list(installed) == ["my-speaker"]
    assert not (installed["my-speaker"] / ".DS_Store").exists()


def test_a_zip_with_several_packs_installs_all_of_them(tmp_path):
    write_pack(tmp_path / "src", "one")
    write_pack(tmp_path / "src", "two")
    path = tmp_path / "both.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("one", "two"):
            archive.write(tmp_path / "src" / name / "voice.npz", f"{name}/voice.npz")
    assert sorted(install_voice_zip(path, tmp_path / "home")) == ["one", "two"]


def test_a_zip_with_no_pack_in_it_says_so(tmp_path):
    path = tmp_path / "holiday.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("photo.jpg", b"not a voice")
    with pytest.raises(FileNotFoundError, match="no voice pack"):
        install_voice_zip(path, tmp_path / "home")


def test_a_file_that_is_not_a_zip_says_so(tmp_path):
    path = tmp_path / "voice.zip"
    path.write_bytes(b"definitely not a zip")
    with pytest.raises(ValueError, match="not a zip"):
        install_voice_zip(path, tmp_path / "home")


def test_a_member_path_cannot_escape_the_install_directory(tmp_path):
    """Members are written by name into a directory we choose, so the classic
    `../../` archive cannot reach outside it."""
    home = tmp_path / "home"
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(write_pack(tmp_path / "src", "ok") / "voice.npz",
                      "../../escaped/voice.npz")
    installed = install_voice_zip(path, home)
    for vdir in installed.values():
        assert home in vdir.parents
    assert not (tmp_path.parent / "escaped").exists()


# ── through the model ────────────────────────────────────────────────────────

@pytest.mark.model
def test_add_voices_takes_the_downloaded_zip(tts, tmp_path, monkeypatch):
    monkeypatch.setattr("zerotts.voices.USER_VOICES_DIR", tmp_path / "home")
    path = zip_pack(tmp_path, "downloaded", n_queries=tts.n_voice_queries)
    assert tts.add_voices(path) == ["downloaded"]
    assert "downloaded" in tts.list_voices()
    # Reachable by name everywhere a bundled voice is.
    assert tts.resolve_voice("downloaded").shape[1] == tts.n_voice_queries


@pytest.mark.model
def test_add_voices_still_takes_a_directory(tts, tmp_path):
    vdir = write_pack(tmp_path, "downloaded", n_queries=tts.n_voice_queries)
    assert tts.add_voices(vdir) == ["downloaded"]
    assert "downloaded" in tts.list_voices()


@pytest.mark.model
def test_a_zip_for_other_weights_is_rejected_and_not_left_installed(tts, tmp_path,
                                                                   monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr("zerotts.voices.USER_VOICES_DIR", home)
    path = zip_pack(tmp_path, "wrong", n_queries=tts.n_voice_queries + 1)
    with pytest.raises(ValueError, match="different weights"):
        tts.add_voices(path)
    assert "wrong" not in tts.list_voices()
    # And it must not come back on the next run.
    assert installed_voices(home) == {}
