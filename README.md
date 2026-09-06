# personalized-voice-isolation

A real-time microphone filter that passes your voice and suppresses everything
else: background noise, room reverb, and other people talking. Output routes to
a virtual audio device, so Discord and anything else that takes a microphone can
use it.

You record a short enrollment, which becomes a speaker embedding. Everything
downstream is conditioned on that embedding.

## Where this actually is

Shipped is a speaker-identity gate. It mutes the microphone during intervals
when you are not the one talking. It does not separate voices: when somebody
talks at the same time as you, the gate is open and both voices pass through.

Handling that overlap needs a speaker-conditioned mask over the mixture, not a
gate. `probe.py` tests whether a pretrained extraction model can do that on
audio from your own room, offline, before anyone builds a realtime one.

On the author's setup it can. With two people talking over each other at
similar volume, the audio the model discards scores 0.602 against the
interferer's voiceprint and 0.063 against the enrolled speaker's: it is
removing the right person. That is one room on one day, and the model tested is
non-causal, runs at 8 kHz, and is far slower than real time, so it is evidence
that the approach transfers and not a working realtime filter. Run the probe on
your own recordings rather than taking this on trust.

Component timings are in `docs/PLAN.md`. End-to-end latency has not been
measured and there is no realtime mask estimator yet.

## Install

Requires Python 3.11 and a virtual audio device (BlackHole on macOS, VB-Cable
on Windows).

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Model weights, about 120 MB in total, download on first use into `models/`.

## Use

```
python enroll.py --list-devices
python enroll.py --name <you> --device <mic> --clips 8 --impostors 4
```

Enroll on the microphone and in the room you will actually use. This writes
`speakers.npz` (centroids and a threshold) and `clips/<you>/*.wav` (reference
recordings, which `probe.py` needs later).

```
python live.py --list-devices
python live.py --in <mic> --out <cable> --monitor
```

`--monitor` prints the live similarity score. Watch it while you and someone
else take turns talking, and adjust `--threshold` until it separates you
cleanly. The value enrollment suggests is a starting point, not a calibration.

```
python live.py --in <mic> --out <cable> --record-debug session.wav
python probe.py --debug-wav session.wav
```

Record a couple of minutes including deliberate cross-talk, then probe it and
listen to the files in `probe_out/`. `probe.py`'s docstring explains how to read
the result:

```
python -c "import probe; print(probe.__doc__)"
```

## Layout

The four scripts at the root are thin entry points. The code lives in `pvi/`.

```
enroll.py  live.py  probe.py  bench.py     entry points
pvi/
  dsp/        shared signal path: constants, resampling, embeddings, denoiser
  enroll/     recording, clip analysis, threshold, speakers.npz
  live/       ring buffer, realtime pipeline, devices, guided session
  tse/        extraction backends and their licences
  probe/      offline extraction test and its reporting
  bench/      timing against the audio callback budget
docs/PLAN.md            where this is going, phase by phase
.claude/CLAUDE.md       invariants that must not be broken
```

Everything imports `pvi.dsp` and nothing reimplements any part of it. Enrollment
and inference drifting apart is this project's classic silent bug: it raises
nothing and quietly invalidates every threshold.

`speakers.npz` holds enrollment centroids. They are biometric data, and they are
gitignored. So are `models/`, `clips/` and every `.wav`.

## Licence

MIT, see LICENSE. This does not extend to model weights downloaded at runtime.
TD-SpeakerBeam in particular, which `probe.py` fetches, ships under a BUT/NTT
evaluation-only licence: free to use internally for testing and evaluation, not
redistributable and not modifiable. Running the probe is inside that grant.
Shipping anything built on those weights is not.
