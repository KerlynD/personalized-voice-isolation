# CLAUDE.md

## The goal

Close Enough Personalized speech enhancement (PSE): a real-time microphone filter that
passes the enrolled speaker's voice and suppresses everything else, background
noise, room reverb, **and other people talking, including when they talk at the
same time as the enrolled speaker.** Output routes to a virtual audio device so
Discord and similar apps can use it as a mic.

The concurrent-speech case is the point of the project. A filter that only works
when the target is silent does not solve the problem.

## Where the project actually is

**Shipped (v0.1): a speaker-identity gate.** Untested, but it is a *gate*, it
mutes intervals when the enrolled speaker is not talking. When someone talks
over the enrolled speaker, the gate is open and both voices pass. This is a
known, documented limitation, not a bug to fix incrementally. Gating cannot
become separation.

**The actual target: a speaker-conditioned mask estimator.** Same ECAPA
embedding, but instead of driving a binary gain it conditions a model that
estimates a time-frequency mask over the mixture, frame by frame. That is what
suppresses an interferer whose speech overlaps the target's.

**Immediate open question, unanswered:** do existing pretrained TSE models
transfer zero-shot to this acoustic condition? They are trained on two speakers
at roughly equal loudness. This setup is one loud close-mic speaker and one
quiet reverberant interferer at roughly +15 to +25 dB SNR. Nobody has measured
this. Answering it decides whether the remaining work is streaming engineering
or model fine-tuning. **Do not start either until the probe is run.**

## Repo layout

The four scripts at the root are thin entry points into `pvi/`.

- `pvi/dsp/` — shared signal path: constants, resampling, embeddings, denoiser
- `pvi/enroll/` — recording, clip analysis, threshold, speakers.npz
- `pvi/live/` — ring buffer, realtime pipeline, devices, guided session
- `pvi/tse/` — extraction backends and their licences
- `pvi/probe/` — offline extraction test and its reporting
- `pvi/bench/` — timing against the callback budget
- `docs/PLAN.md` — the phased plan from here to a shipped app

Everything imports `pvi.dsp` and nothing reimplements any part of it. Each
package keeps its tunable constants at the top of the module they belong to,
with a comment explaining the tradeoff, and its CLI in `cli.py` doing
orchestration only.

The realtime harness (48 kHz capture, virtual device routing, non-blocking
callback, ring buffer, resampling, enrollment, debug capture) is reusable as-is
for PSE. Only the analysis thread's decision logic gets replaced.

## Invariants — do not break these

**1. Enrollment and inference must share one preprocessing path.**
Both go through `Denoiser` then `dsp.to_analysis()` then `dsp.embed()`.
Divergence here raises no error — it silently shifts similarity scores and
looks like a threshold problem.

**2. The audio callback must never block.**
No file I/O, no model loading, no locks. It runs every 10.67 ms. DeepFilterNet
inference is fine there (~1.5 ms/frame). ECAPA is not — it lives in the
analysis thread. Any PSE model that goes in the callback must be benchmarked
against that budget first.

**3. Gain changes are always ramped.** `RAMP_MS` exists because stepping gain
between blocks clicks audibly.

**4. Closed-gate gain is `FLOOR`, not zero.** Dead silence sounds like a
dropped connection.

**5. Sample rates are load-bearing.** Audio path 48 kHz (`dsp.SR`), ECAPA
requires 16 kHz (`dsp.ANALYSIS_SR`), `dsp.FRAME` is 512 because that is
DeepFilterNet's native frame at 48 kHz. Do not unify them.

**6. Latency budget is ~50 ms of added algorithmic delay.** Discord already
costs 100–150 ms one-way. Past roughly 250 ms total, turn-taking feels broken.
Any PSE model must be causal or use short lookahead. Most published TSE models
are non-causal and cannot be used as-is.

## Things that are commonly misunderstood here

- **DeepFilterNet does not remove other people's voices.** It is a noise
  suppressor trained to *preserve* speech. Fans, keyboards, HVAC, reverb, yes.
  Interfering talkers, no. Never describe it as separation.
- **The gate is not enhancement.** Do not describe v0.1 as isolating the user's
  voice. It mutes other people's turns. Say that plainly in user-facing docs.
- **The threshold is a cosine similarity**, roughly 0.2–0.8 in practice. Not dB,
  not a probability, not normalized per-user. Calibrated per mic and room.
- **PSE ducks interferers, it does not delete them.** Single-channel separation
  makes a competing voice quiet and unintelligible. Expected outcome is
  "murmur under my voice," not silence. Do not write docs implying otherwise.

## Relevant prior art

- VoiceFilter-Lite — the reference design for *streaming* speaker-conditioned
  masking. Read this for the architecture pattern.
- pDCCRN — the DNS Challenge personalized-DNS track baseline. Concrete,
  reproducible recipe for this exact task.
- TD-SpeakerBeam (`BUTSpeechFIT/speakerbeam`) — released weights, readable
  code, non-causal. The candidate for the zero-shot probe.
- `microsoft/DNS-Challenge` — public dataset with per-speaker enrollment clips,
  structured for training PSE models.

## Conventions

- Comments explain *why*. This repo teaches how speaker embeddings work; the
  docstrings are part of the product.
- No emoji in code, comments, docs, or commit messages.
- No em-dashes.
- 
- Keep dependencies small. Every one is a barrier for someone running this on
  their own machine.
- Tunable constants live at module top with a comment explaining the tradeoff.
  No magic numbers buried in function bodies.
- Hardware available for training: single RTX 5070 Ti, 16 GB. Size proposals
  accordingly — fine-tuning released weights, not training from scratch.

## Never commit

- `speakers.npz` — enrollment centroids are biometric data
- `models/` — downloaded weights
- `*.wav` — debug recordings contain real speech

## What has and has not been measured

Measured, on real recordings from the target room:

- Zero-shot TSE transfer. See "Where the project actually is" for the numbers
  and the caveats. Two sessions, one room, one day.
- Enrollment separability. A clean enrollment gave a +0.263 margin between the
  enrolled speaker's 5th-percentile window and the interferer's 95th. A first
  attempt gave -0.111, from a clip with a second voice audible on it plus a
  threshold rule that keyed off a single worst window.

Timing, from `python -m pvi.bench` on an M-series Mac at one torch thread. The callback
budget is 10.67 ms.

    DeepFilterNet, one frame        0.30 ms p50, 0.74 ms max      7% of budget
    ECAPA, one 1.5 s window        13.50 ms      (analysis thread, 200 ms apart)
    TSE extractor, one call        ~24 ms        225% of budget

The extractor number is nearly independent of how much audio you give it: 10.7
ms, 32 ms and 64 ms chunks all cost about the same. It is 130 Conv1d layers at
roughly 190 us each, so it is PyTorch per-op dispatch and not arithmetic. More
threads make it worse; torch.jit.trace buys 6%. DeepFilterNet is a comparable
model running 100x faster through ONNX Runtime on the same machine, which is
where the headroom is.

Encoding the enrollment costs 37 ms per 3 s and scales linearly, and pvi.probe
pays it on every chunk. It produces a fixed 512-dim vector and is entirely
cacheable.

**The gate is not redundant.** On the interferer-only phase of a real
recording, the extractor attenuates by 1.1 dB where the gate attenuates by 34.
TD-SpeakerBeam was trained on Libri2Mix, where the target is present in every
mixture, so it has never been taught what to do when the target is silent. The
mask estimator handles overlap; the gate handles absence. They are different
problems and the product needs both.

Not measured, and not to be claimed:

- Gate accuracy in use. No false-reject or leak rate has been counted.
- Anything about a causal or 48 kHz model, which does not exist yet.
- Any other room, microphone, or interferer.
- End-to-end latency. Only per-component compute has been timed.

Do not write performance claims into the README beyond the list above, and do
not assert that an approach works or fails until it has been run on real
recordings from the target room.