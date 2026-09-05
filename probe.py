#!/usr/bin/env python3
"""
probe.py - does a pretrained mask estimator work on this room, before we build one?

The gate in live.py mutes other people's turns. It cannot touch a voice that
overlaps yours, because a gate decides when to pass audio and not which parts of
it to pass. Fixing that means a speaker-conditioned mask estimator, and there
are two very different ways to get one:

    streaming engineering  take a model that already works on this audio and
                           make it causal, 48 kHz, and callback-sized
    fine-tuning            take a model that does not work on this audio and
                           retrain it on data that matches the room

Published target-speech-extraction models are trained on two speakers at roughly
equal loudness. This setup is one loud close-mic speaker plus one quiet
reverberant interferer, somewhere around +15 to +25 dB SNR. Whether a model
crosses that gap decides which of the two projects you are actually starting.
Nobody has measured it, so this script measures it.

    enroll.py   ->  speakers.npz + clips/<name>/*.wav
    live.py     ->  session.wav   (--record-debug, L=raw mic, R=gated output)
    probe.py    ->  probe_out/*.wav, to listen to

HOW IT WORKS
------------
1. Clone TD-SpeakerBeam into models/ and load the published checkpoint.
2. Read the raw mic channel of a live.py debug capture, decimate 48k -> 8k.
3. Read your enrollment clips, concatenate, decimate the same way.
4. RMS-normalize both to roughly the level the model trained at, so that a
   negative result means "wrong acoustic condition" and not "wrong level".
5. Run the model over the mixture, conditioned on the enrollment, in
   crossfaded chunks.
6. Write the input, the output, and the v0.1 gate side by side at one shared
   gain, so their relative loudness is honest.

THE MODEL
---------
TD-SpeakerBeam (Zmolikova et al., BUT / NTT), the checkpoint published in
BUTSpeechFIT/speakerbeam at example/model.pth. Chosen because it has released
weights, readable code, and the exact interface we need: a mixture waveform plus
an enrollment waveform in, one extracted waveform out.

Its training conditions, read straight out of the checkpoint's model_args:

    sample_rate       8000        band-limited to 4 kHz, see the note below
    causal            False       needs the whole file, ~0 ms is not on offer
    mask_act          relu        unbounded mask, output can exceed the input
    i_adapt_layer     7           speaker conditioning enters at TCN block 7
    adapt_enroll_dim  128         its own learned embedding, NOT ECAPA

Trained on Libri2Mix at 8 kHz: two speakers, near-equal loudness, task
sep_noisy. That is the mismatch we are measuring.

None of those properties are things the realtime path could adopt. This script
is offline, non-causal, 8 kHz, and many times slower than real time, and that is
fine, because it is a listening test rather than a prototype. Nothing here
should leak into live.py.

WHY speakers.npz CANNOT DRIVE IT
--------------------------------
TD-SpeakerBeam learned its own 128-dim speaker representation jointly with its
extraction network. ECAPA's 192-dim VoxCeleb space is a different coordinate
system and there is no mapping between them. So the enrollment reference has to
be a WAVEFORM, which is why enroll.py saves clips/<name>/*.wav. speakers.npz is
used here only for the similarity readout, never for conditioning.

LICENCE WARNING
---------------
The TD-SpeakerBeam code and checkpoint are NOT open source. They ship under a
BUT/NTT "Software License Agreement for Evaluation" which grants a royalty-free
licence to use the software internally for testing, analyzing and evaluating,
and explicitly forbids redistribution, modification, and transfer to third
parties. Running this probe is squarely inside that grant. Vendoring the code or
the weights into this MIT-licensed repository is not, and shipping anything
derived from those weights is not either. That is why this script clones the
upstream repo at runtime instead of checking anything in, and why models/ is
gitignored. If the probe succeeds, the follow-up work needs either a permissively
licensed model or a clean-room implementation trained on DNS-Challenge data.

READING THE RESULT
------------------
Compare _extracted against _mix, never against the original 48 kHz recording.
Both have been through the same decimation to 8 kHz, so anything you hear
between them is the model. Against the 48 kHz original you will hear everything
above 4 kHz missing and read it as extraction damage. It is not. It is the
checkpoint's sample rate.

Find a stretch where both people are talking at once. That stretch is the whole
experiment. Every stretch where only one person talks tells you nothing you did
not already know from the gate.

Then judge one of four outcomes:

  Interferer ducked, target intact.
      The other voice drops to a murmur under yours and your own speech comes
      through unchanged. This is the success case, and "murmur" really is the
      ceiling: single-channel extraction makes a competing voice quiet and
      unintelligible, it does not delete it. Expect to still hear something.
      The next question becomes latency, not modelling.

  Interferer ducked, target damaged.
      The other voice drops but yours acquires warble, lisping, or dropouts.
      The conditioning works and the model is fighting the level mismatch.
      Points at fine-tuning on matched data, not at a different architecture.

  Nothing ducked.
      Output sounds like the input. The speaker conditioning is not engaging at
      all, most likely because at +20 dB SNR the model reads the mixture as
      already being a single speaker. This is the informative negative, and it
      means fine-tuning on matched-SNR data rather than probing more checkpoints.

  Wrong speaker extracted.
      The interferer comes through and you are suppressed. Check the enrollment
      first: a clip that is too short, or one with two voices in it, will do
      this before the model is at fault.

Whatever you conclude, it is one recording in one room. Record a second session
on a different day before treating the answer as settled.

Usage:
    python probe.py --debug-wav session.wav
    python probe.py --debug-wav session.wav --speaker angel --start 40 --dur 30
    python probe.py --debug-wav session.wav --enroll-wav some_other_clip.wav
"""

import argparse
import pathlib
import subprocess
import sys
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import dsp

# --- tunables -------------------------------------------------------------

# TD-SpeakerBeam's training rate. Not adjustable: the filterbank kernel size
# (16 samples) and stride (8) are learned at this rate, so feeding it 16 kHz
# would halve the effective analysis window and put the model well outside
# anything it has seen. Band-limiting to 4 kHz is the cost of using released
# weights, and it is a property of this checkpoint rather than of PSE.
PROBE_SR = 8000

# Whole-file inference on a ConvTasNet-shaped network allocates activations
# proportional to clip length: roughly 512 channels x (samples / 8) frames x 24
# blocks. A three-minute capture will exhaust memory on a laptop. Chunking is a
# memory concession, not part of the method. Chunks are crossfaded rather than
# butt-joined because gLN normalizes over the whole sequence, so two chunks
# processed independently land at slightly different output levels and a hard
# splice steps the gain audibly, for the same reason RAMP_MS exists in live.py.
# Set --chunk-sec 0 to process the whole file at once if you have the RAM.
CHUNK_SEC = 20.0
CHUNK_OVERLAP_SEC = 1.0

# The model was trained on Libri2Mix, which is loudness-normalized. A close-mic
# capture is far hotter than that. Normalizing input RMS to roughly what the
# model saw in training removes one confound from the experiment, so that a
# negative result means "wrong acoustic condition" rather than "wrong level".
TARGET_RMS = 0.03

# The auxiliary network averages over the whole enrollment, so longer is
# generally better, but it was trained with 3-second segments. Below the floor
# the speaker embedding is too noisy for the result to mean anything; above the
# ceiling you are spending memory for no additional speaker information.
MIN_ENROLL_SEC = 2.0
MAX_ENROLL_SEC = 20.0

# Peak level of the written files. Left below 1.0 so that a relu mask, which is
# unbounded and can overshoot the mixture, does not clip on the way to disk.
OUT_PEAK = 0.89

SPEAKERBEAM_REPO = "https://github.com/BUTSpeechFIT/speakerbeam.git"
SPEAKERBEAM_DIR = dsp.MODEL_DIR / "speakerbeam"


# --- upstream checkout ----------------------------------------------------

def ensure_speakerbeam(root=SPEAKERBEAM_DIR):
    """Clone TD-SpeakerBeam into models/ and put its src/ on sys.path.

    Cloned at runtime rather than vendored, for the licence reason in the module
    docstring. models/ is gitignored, so nothing from upstream can be committed
    here by accident.

    The upstream modules import each other as `from models.base_models_informed
    import ...`, with no package prefix, so src/ itself has to be the sys.path
    entry. That collides with our own models/ directory name only on disk, not
    in the import namespace, because we never import our models/ as a package.
    """
    root = pathlib.Path(root)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning TD-SpeakerBeam into {root} (about 30 MB)")
        subprocess.run(
            ["git", "clone", "--depth", "1", SPEAKERBEAM_REPO, str(root)],
            check=True,
        )

    ckpt = root / "example" / "model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{ckpt} is missing. The checkpoint lives in the upstream repo under "
            f"example/. Delete {root} and re-run to fetch it again."
        )

    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return ckpt


def load_speakerbeam(ckpt_path, device="cpu"):
    """Build TimeDomainSpeakerBeam and load the published weights.

    Deliberately does NOT use asteroid's `Model.from_pretrained`, which is what
    the upstream demo notebook calls. Two reasons, both of which are first-run
    crashes on a current environment:

      1. asteroid 0.7.0 calls `torch.load(path, map_location="cpu")` with no
         `weights_only` argument. Since torch 2.6 that argument defaults to
         True, and the load fails on anything the allowlist does not recognise.
      2. It routes the path through `huggingface_hub.cached_download`, which
         newer huggingface-hub releases removed entirely.

    Doing the two steps by hand skips both. `weights_only=True` is tried first
    and is expected to succeed: this checkpoint's pickle references only
    OrderedDict, torch.FloatStorage and torch._utils._rebuild_tensor_v2, all of
    which are on torch's allowlist. The fallback exists for other checkpoints.
    """
    import torch
    from models.td_speakerbeam import TimeDomainSpeakerBeam

    try:
        conf = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"note: safe load failed ({type(e).__name__}), retrying unrestricted")
        print("      only do this for checkpoints you trust the origin of")
        conf = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    for key in ("model_args", "state_dict"):
        if key not in conf:
            raise ValueError(f"checkpoint has no '{key}' key; got {list(conf)}")

    model = TimeDomainSpeakerBeam(**conf["model_args"])
    model.load_state_dict(conf["state_dict"])
    model.eval().to(device)

    sr = int(conf["model_args"].get("sample_rate", PROBE_SR))
    if sr != PROBE_SR:
        raise ValueError(
            f"checkpoint sample_rate is {sr}, probe is built around {PROBE_SR}"
        )
    return model


# --- audio helpers --------------------------------------------------------

def resample_to(x, sr_in, sr_out):
    """Rational resample with a proper anti-alias filter.

    Same reasoning as dsp.to_analysis: whole-clip, so there is no filter state
    to carry and no edge artifact at block boundaries. 48000 -> 8000 reduces to
    exactly 1/6, so no approximation is involved on the normal path.
    """
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    r = Fraction(sr_out, sr_in).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator).astype(np.float32)


def read_debug_wav(path):
    """Read a live.py --record-debug capture. Returns (raw, gated, sr).

    The capture is stereo by contract: L is the raw mic before any gain, R is
    what actually went to the virtual device. `gated` is None for a mono file,
    which is accepted so that an ordinary recording can be probed too.
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        print(f"note: {path} is mono; treating it as the raw channel, no gate to compare")
        return x[:, 0], None, sr
    if x.shape[1] > 2:
        print(f"note: {path} has {x.shape[1]} channels; using 0 as raw and 1 as gated")
    return x[:, 0], x[:, 1], sr


def find_enrollment(clip_dir, speaker):
    """Collect the reference clips enroll.py saved for this speaker."""
    d = pathlib.Path(clip_dir) / speaker
    clips = sorted(d.glob("*.wav"))
    if not clips:
        raise SystemExit(
            f"no enrollment clips in {d}.\n"
            f"probe.py needs a waveform of {speaker} speaking alone, because "
            f"TD-SpeakerBeam cannot use an ECAPA vector. Either re-run "
            f"enroll.py, which saves clips there, or pass --enroll-wav "
            f"pointing at any clean solo recording."
        )
    return clips


def read_enrollment(paths):
    """Read reference clips, downmix, concatenate, return at PROBE_SR.

    Concatenated rather than picked one at a time because the auxiliary network
    averages its output over time: more material means a steadier speaker
    embedding, and enroll.py deliberately varies delivery across clips, so using
    all of them covers more of how you actually sound.
    """
    parts = []
    total = 0.0
    for p in paths:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = resample_to(x.mean(axis=1), sr, PROBE_SR)
        parts.append(x)
        total += len(x) / PROBE_SR
        if total >= MAX_ENROLL_SEC:
            break
    x = np.concatenate(parts)[: int(MAX_ENROLL_SEC * PROBE_SR)]
    used = len(parts)

    secs = len(x) / PROBE_SR
    if secs < MIN_ENROLL_SEC:
        print(
            f"warning: enrollment is {secs:.1f}s, under the {MIN_ENROLL_SEC}s floor. "
            f"The speaker embedding will be noisy and a poor result may say more "
            f"about the enrollment than about the model."
        )
    return x, used


def rms_normalize(x, target=TARGET_RMS):
    """Scale to a target RMS. Returns (scaled, gain) so it can be undone."""
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    g = target / rms
    return (x * g).astype(np.float32), g


# --- extraction -----------------------------------------------------------

def extract(model, mix, enroll, chunk_sec=CHUNK_SEC, overlap_sec=CHUNK_OVERLAP_SEC):
    """Run TD-SpeakerBeam over `mix`, conditioned on `enroll`. Both at PROBE_SR.

    The enrollment is passed to every chunk rather than being embedded once and
    reused. The auxiliary network is deterministic and sees identical input each
    time, so the embedding is identical; recomputing it costs a little time and
    saves reaching into model internals that upstream may reorganise.
    """
    import torch

    enroll_t = torch.from_numpy(np.ascontiguousarray(enroll)).float().unsqueeze(0)

    def run(seg):
        seg_t = torch.from_numpy(np.ascontiguousarray(seg)).float().unsqueeze(0)
        with torch.no_grad():
            y = model(seg_t, enroll_t)
        # (batch, 1, time) for a 2D input. Squeeze back to 1D.
        return y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    n = len(mix)
    if chunk_sec <= 0 or n <= int(chunk_sec * PROBE_SR):
        return run(mix)[:n]

    chunk = int(chunk_sec * PROBE_SR)
    overlap = max(1, int(overlap_sec * PROBE_SR))
    hop = chunk - overlap
    if hop <= 0:
        raise ValueError("chunk_sec must be larger than overlap_sec")

    # Build the chunk starts up front so the tail can be folded into the last
    # chunk. A stub final chunk of a few hundred samples is shorter than the
    # network's receptive field and produces garbage.
    starts = list(range(0, max(1, n - overlap), hop))
    while len(starts) > 1 and n - starts[-1] < overlap * 2:
        starts.pop()

    out = np.zeros(n, dtype=np.float32)
    fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    for i, s in enumerate(starts):
        e = n if i == len(starts) - 1 else min(n, s + chunk)
        y = run(mix[s:e])[: e - s]
        print(f"  chunk {i + 1}/{len(starts)}  {s / PROBE_SR:6.1f}s - {e / PROBE_SR:6.1f}s")
        if i == 0:
            out[s:e] = y
            continue
        k = min(overlap, len(y))
        out[s:s + k] = out[s:s + k] * (1.0 - fade[:k]) + y[:k] * fade[:k]
        out[s + k:e] = y[k:]
    return out


# --- optional ECAPA scoring ----------------------------------------------

def load_centroids(path):
    """Load enrollment centroids from speakers.npz. Returns (names, matrix).

    The format enroll.py writes: 'names', 'centroids', 'threshold'.
    """
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    mat = np.asarray(z["centroids"], dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != dsp.EMB_DIM:
        raise ValueError(
            f"{path} centroids have shape {mat.shape}, expected (n, {dsp.EMB_DIM})"
        )
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return names, mat.astype(np.float32)


def score_against(encoder, centroids, wav8):
    """Cosine similarity of an 8 kHz clip against the enrolled centroids.

    READ THIS BEFORE TRUSTING THE NUMBER.

    These scores are comparable to each other and to nothing else. They are NOT
    comparable to live.py's threshold, and you must not calibrate the gate from
    them. CLAUDE.md invariant 1 says enrollment and inference share one
    preprocessing path: Denoiser, then dsp.to_analysis, then dsp.embed. This
    path is different on both counts. The audio here has been decimated to 8 kHz
    and upsampled back to 16 kHz, so everything above 4 kHz is gone and ECAPA
    sees a spectrum it was not enrolled on; and the probe deliberately does not
    denoise, because inserting DeepFilterNet ahead of the extractor would
    confound the one thing this script is measuring.

    What the number is good for is the delta. If extraction raised similarity to
    the enrolled speaker relative to the mixture, the model moved the signal
    toward the target. That direction is meaningful even though the absolute
    value is not. Treat it as a sanity check on your ears, not as a measurement.
    """
    wav16 = resample_to(wav8, PROBE_SR, dsp.ANALYSIS_SR)
    emb = dsp.embed(encoder, wav16)
    return centroids @ emb


# --- main -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Zero-shot TD-SpeakerBeam probe on a real capture. "
                    "Offline, non-causal, 8 kHz.",
    )
    p.add_argument("--debug-wav", required=True,
                   help="stereo capture from live.py --record-debug (L=raw, R=gated)")
    p.add_argument("--speakers", default="speakers.npz")
    p.add_argument("--speaker", default=None,
                   help="which enrolled speaker to extract; defaults to the first")
    p.add_argument("--clip-dir", default="clips",
                   help="where enroll.py saved the reference recordings")
    p.add_argument("--enroll-wav", default=None,
                   help="override the reference clips with one specific file")
    p.add_argument("--out-dir", default="probe_out")
    p.add_argument("--start", type=float, default=0.0, help="seconds into the capture")
    p.add_argument("--dur", type=float, default=0.0, help="seconds to process, 0 for all")
    p.add_argument("--chunk-sec", type=float, default=CHUNK_SEC,
                   help="0 processes the whole file at once; needs more RAM")
    p.add_argument("--device", default="cpu", help="cpu, or cuda if you have it")
    a = p.parse_args(argv)

    dsp.limit_torch_threads()

    names, cents = None, None
    if pathlib.Path(a.speakers).exists():
        names, cents = load_centroids(a.speakers)
    speaker = a.speaker or (names[0] if names else None)
    if speaker is None and a.enroll_wav is None:
        raise SystemExit(
            f"no {a.speakers} and no --enroll-wav. Run enroll.py first, or pass "
            f"--enroll-wav pointing at a clean solo recording."
        )
    if names and speaker not in names:
        raise SystemExit(f"{speaker!r} is not enrolled in {a.speakers}: {names}")

    ckpt = ensure_speakerbeam()

    raw48, gated48, sr = read_debug_wav(a.debug_wav)
    if sr != dsp.SR:
        print(f"note: capture is {sr} Hz, not dsp.SR ({dsp.SR}); resampling anyway")

    # Trim before resampling so the requested window is exact in source samples.
    i0 = int(a.start * sr)
    i1 = len(raw48) if a.dur <= 0 else min(len(raw48), i0 + int(a.dur * sr))
    if i0 >= len(raw48):
        raise SystemExit(f"--start {a.start}s is past the end of a {len(raw48) / sr:.1f}s file")
    raw48 = raw48[i0:i1]
    if gated48 is not None:
        gated48 = gated48[i0:i1]

    mix = resample_to(raw48, sr, PROBE_SR)
    clips = [pathlib.Path(a.enroll_wav)] if a.enroll_wav else find_enrollment(a.clip_dir, speaker)
    enroll, n_used = read_enrollment(clips)
    print(f"target     {speaker or '(from --enroll-wav)'}")
    print(f"mixture    {len(mix) / PROBE_SR:.1f}s at {PROBE_SR} Hz")
    print(f"enrollment {len(enroll) / PROBE_SR:.1f}s from {n_used} of "
          f"{len(clips)} clip(s)")

    # Normalize both to the level the model trained at. The same gain is undone
    # on the output so the written files stay comparable to the input.
    mix_n, mix_gain = rms_normalize(mix)
    enroll_n, _ = rms_normalize(enroll)

    model = load_speakerbeam(ckpt, device=a.device)
    print("extracting")
    est_n = extract(model, mix_n, enroll_n, chunk_sec=a.chunk_sec)
    est = est_n / mix_gain

    # One shared gain across every written file. Per-file peak normalization
    # would hide exactly what the probe is looking for: whether the interferer
    # got quieter relative to the target.
    peak = max(float(np.abs(mix).max()), float(np.abs(est).max()), 1e-9)
    g = OUT_PEAK / peak
    if g < 1.0:
        print(f"note: scaling all outputs by {g:.2f} to fit; relative levels preserved")

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(a.debug_wav).stem
    written = []
    for tag, sig in (("mix", mix), ("extracted", est)):
        f = out_dir / f"{stem}_{tag}.wav"
        sf.write(f, (sig * g).astype(np.float32), PROBE_SR)
        written.append(f)
    if gated48 is not None:
        f = out_dir / f"{stem}_gate.wav"
        sf.write(f, (resample_to(gated48, sr, PROBE_SR) * g).astype(np.float32), PROBE_SR)
        written.append(f)

    print("\nwrote:")
    for f in written:
        print(f"  {f}")

    if cents is not None:
        print("\nECAPA similarity (relative only, see score_against docstring):")
        encoder = dsp.load_encoder(device=a.device)
        s_mix = score_against(encoder, cents, mix)
        s_est = score_against(encoder, cents, est)
        print(f"  {'speaker':<20} {'mixture':>8} {'extracted':>10} {'delta':>8}")
        for i, n in enumerate(names):
            print(f"  {n:<20} {s_mix[i]:8.3f} {s_est[i]:10.3f} {s_est[i] - s_mix[i]:+8.3f}")

    print("\nListen to _extracted against _mix, not against the 48 kHz original.")
    print("Find a stretch where two people talk at once; that is the experiment.")
    print("The four outcomes and what each one implies are in this file's docstring:")
    print("  python -c \"import probe; print(probe.__doc__)\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
