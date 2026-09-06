"""Capturing reference clips from a microphone, and writing them to disk."""

import pathlib

import numpy as np
import sounddevice as sd
import soundfile as sf

from .. import dsp

# Cycled through the target's clips so the voiceprint covers a range of
# deliveries rather than eight takes of the same sentence at the same volume.
# A voiceprint built from identical clips only recognises you when you sound
# exactly like that, which is never, on a real call.
DELIVERY = [
    "normal speaking voice, how you sound on a call",
    "a bit quieter, like it is late and someone is asleep",
    "louder and more animated, like you are excited about something",
    "normal again, but lean back away from the mic",
    "normal, but lean in close to the mic",
    "faster, like you are telling a story you are into",
    "slower and lower, like you are explaining something carefully",
    "normal speaking voice again",
]

TARGET_INSTRUCTIONS = [
    "Only you. Nobody else in the room may speak, at any volume.",
    "Talk continuously for the whole time: read something out loud,",
    "describe your day, count. Do not pause for more than a second.",
]

IMPOSTOR_INSTRUCTIONS = [
    "The OTHER person talks. You stay completely silent.",
    "They sit where they normally sit, at their normal distance.",
    "Have them talk at a NORMAL to LOUD level, not quietly, and",
    "keep talking for the whole clip. The cutoff is calibrated",
    "against how loud they actually get, so a quiet clip sets it",
    "too permissive and they leak through when they get animated.",
]


def record(seconds, header, instructions, device=None, channel=0):
    """Prompt, then capture `seconds` of one channel of `device`.

    sd.rec's `mapping` counts channels from 1 the way macOS does, while the
    rest of this project counts from 0 the way the arrays do, hence the +1.
    Selecting a channel matters on an Aggregate Device, where the members'
    channels stack end to end and the microphone is often not channel 0.
    """
    print(f"\n{header}")
    for line in instructions:
        print(f"    {line}")
    input(f"  [Enter] to start recording {seconds:.0f}s > ")
    buf = sd.rec(int(seconds * dsp.SR), samplerate=dsp.SR,
                 dtype="float32", device=device, mapping=[channel + 1])
    sd.wait()
    wav = np.ascontiguousarray(buf[:, 0])
    peak = float(np.max(np.abs(wav)))
    level = dsp.rms(wav)
    print(f"  rms={level:.4f} peak={peak:.3f}")
    if level < 0.005:
        print("  WARNING: very quiet. Check input device and gain.")
    if peak > 0.99:
        print("  WARNING: clipping. Back off the gain or move off-axis.")
    return wav


def save_clip(clip_dir, name, index, wav48):
    """Write the raw recording to clips/<name>/.

    For the enrolled speaker these are pvi.probe's reference clips, and they
    are the reason a probe run needs no second recording session. Impostor
    clips go to clips/_impostors/ for a different reason: without them you
    cannot re-derive a threshold, re-check a margin or diagnose a bad
    enrollment without dragging the other person back to the microphone.

    Saved raw, before denoising, so that whatever consumes them later can make
    its own decision about preprocessing.
    """
    d = pathlib.Path(clip_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{index:02d}.wav"
    sf.write(path, wav48, dsp.SR)
    return path


def stale_clips(clip_dir, name):
    """Clips a previous enrollment left behind.

    A fresh enrollment replaces the voiceprint, so it has to replace these too.
    pvi.probe reads every wav in clips/<name>/, so a leftover clip would
    silently become part of the probe's reference audio: stale data that looks
    like a bad model result.
    """
    out = sorted((pathlib.Path(clip_dir) / name).glob("*.wav"))
    out += sorted((pathlib.Path(clip_dir) / "_impostors").glob("*.wav"))
    return out
