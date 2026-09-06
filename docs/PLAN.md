# Plan

How this goes from a working gate plus an offline probe to a menu bar app you
turn on before a Discord call.

Written to survive a cold start. If you are picking this up with no memory of
the conversation that produced it, read "Where this actually is" first, then
the phase you are on. Every number here was measured on real recordings and
real hardware; nothing is estimated. `.claude/CLAUDE.md` holds the invariants
that must not be broken while executing any of this.

---

## The goal, stated once

A microphone filter that passes one enrolled speaker and suppresses everything
else: background noise, room reverb, **and other people talking, including when
they talk at the same time as the enrolled speaker.**

The output routes to a virtual audio device, so Discord, Zoom, Meet, OBS and
anything else that accepts a microphone can select it with no per-application
work. That is the whole integration story and it does not change.

Shipped as a macOS menu bar app with an on/off toggle, installable by people
who are not the author, which means the app creates the aggregate audio device
rather than asking anyone to open Audio MIDI Setup.

The concurrent-speech case is the point. A filter that only works when the
target is silent does not solve the problem.

---

## Where this actually is

### Established, with measurements

**Zero-shot extraction transfers to real recordings.** A pretrained
speaker-conditioned mask estimator extracts the enrolled speaker from a real
capture in the target room, with no fine-tuning. Measured by scoring every
output against both speakers' ECAPA centroids, on a session where the two
talkers were within about 7 dB of each other:

```
file           target    other
mix             0.501    0.208
extracted       0.480    0.155
removed         0.063    0.602   <- the discarded audio is almost purely
control         0.076    0.631      the interferer
```

Their score fell 0.049 against the target's 0.021, so it removes them 2.3x
harder than it damages you. Conditioning on a different enrollment produces a
substantially different output (correlation +0.399 between the two passes), so
the conditioning is doing real work rather than passing whoever is loudest.

**The licence blocker is resolved.** The original BUT/NTT TD-SpeakerBeam is
evaluation-only and cannot ship. ESPnet publishes its own TD-SpeakerBeam
(Apache-2.0 code, CC-BY-4.0 weights) which separates *better* on the same
recording. `pvi/tse/` keeps both behind one interface. Build on `espnet`.

**Timing, from `python -m pvi.bench`, one torch thread, callback budget 10.67 ms:**

```
DeepFilterNet, one frame        0.30 ms p50, 0.74 ms max      7% of budget
ECAPA, one 1.5 s window        13.50 ms      (analysis thread, 200 ms apart)
TSE extractor, one call        ~24 ms        225% of budget
```

The extractor's cost is nearly independent of how much audio it is given: 10.7
ms, 32 ms and 64 ms chunks all cost about the same. It is 130 Conv1d layers at
roughly 190 us each, so it is PyTorch per-operation dispatch, not arithmetic.
More threads make it worse. `torch.jit.trace` buys 6%. Encoding the enrollment
costs a further 37 ms per 3 s and is entirely cacheable.

**The gate is not redundant.** On the interferer-only phase of a real
recording, the extractor attenuates by 1.1 dB where the gate attenuates by 34.
TD-SpeakerBeam trained on Libri2Mix, where the target is present in every
mixture, so it was never taught what to do when the target is silent. The mask
handles overlap; the gate handles absence. The product needs both.

### Not established, and not to be claimed

- End-to-end latency. Only per-component compute has been timed.
- Gate accuracy in use. No false-reject or leak rate has been counted.
- Anything about a causal or 16 kHz model, because neither exists yet.
- Any room, microphone or interferer other than the one pair tested.

---

## Phase 1 — Make inference fast enough to be possible

**No training. No GPU. This is the phase that decides whether the rest is
worth starting.**

### Why

24 ms per call against a 10.67 ms budget is 225%, and it does not improve by
giving the model less audio. But the arithmetic is negligible: it is framework
dispatch across 130 layers. DeepFilterNet is the existence proof, a
comparable-size model running at 0.30 ms through ONNX Runtime on the same
machine, roughly 100x faster than eager PyTorch on a similar workload.

If ONNX gets the extractor into low single-digit milliseconds, the compute
problem is closed and every remaining problem is a training problem. If it
does not, the architecture is wrong and it is far better to learn that before
spending GPU time training it.

### Steps

1. **Cache the enrollment embedding.** `EspnetTSE.embed_enrollment()` already
   exists and returns a fixed 256-dim vector. Compute it once at enrollment
   time, store it in `speakers.npz` alongside the ECAPA centroid, and feed the
   masker directly. Removes 37 ms per 3 s of reference audio from every call.
   Do this first: it is small, it is pure win, and it simplifies the graph you
   are about to export.
2. **Export the masker to ONNX.** Input is the encoded mixture plus the cached
   speaker embedding; output is the mask. Keep the encoder and decoder out of
   the first export if they complicate it; they are two conv layers and cost
   0.02 ms.
3. **Run it in ONNX Runtime** with `intra_op_num_threads=1`, matching the
   denoiser's configuration.
4. **Verify numerically**, not by eye: the ONNX output must match the PyTorch
   output on the same input to within float tolerance. A silently wrong export
   that still produces plausible speech is the failure mode to guard against.
5. **Re-run `python -m pvi.bench`** and record the new number in `.claude/CLAUDE.md`.

### Success criterion

Under 2 ms per call for a callback-sized chunk. That leaves room for
DeepFilterNet's 0.74 ms worst case inside 10.67 ms with headroom for the
scheduler.

### Risks

- The extractor's forward has complex-tensor branches (`is_complex`) and a
  padding-mask helper that ONNX export can choke on. Mitigation: export the
  TCN alone behind a thin wrapper with those branches removed, since the probe
  path never takes them.
- Dynamic sequence length must stay dynamic in the export, or you get a graph
  that only accepts one chunk size.

### What it unblocks

Everything. Phase 2 is only worth training if the result can run.

---

## Phase 2 — Make the model causal, and 16 kHz

**This is the GPU phase. Hardware: one RTX 5070 Ti, 16 GB.**

### Why causal

The released weights are non-causal in two independent ways, and neither is
fixable by engineering:

- **Global layer norm** normalizes across the entire utterance, so it needs the
  whole recording before it can produce the first sample.
- **The dilated convolutions look ahead**, so every output sample depends on
  future input.

ESPnet's extractor already has a `causal=True` flag which swaps in cumulative
layer norm and causal padding. The architecture supports it; these weights were
simply not trained that way. There is no shortcut: it needs training.

### Why 16 kHz in the same run

8 kHz band-limits the output to 4 kHz, which is telephone quality and is
audibly the biggest remaining limit on how the output sounds. 16 kHz doubles
that to 8 kHz, which is where most of the perceived quality gain lives. Since
the model is being retrained anyway, changing the rate costs one dataset
preparation rather than a second training run.

The end-to-end audio path stays 48 kHz. Only the extractor runs at 16 kHz, with
resampling either side, exactly as ECAPA already runs at 16 kHz today.

### Steps

1. Prepare LibriMix at 16 kHz. ESPnet's `egs2/librimix/tse1` recipe is the
   reference; `microsoft/DNS-Challenge` personalized-DNS data is the fallback
   if LibriMix's two-speaker-at-equal-loudness assumption proves too far from
   the real condition.
2. Fine-tune from the released CC-BY-4.0 checkpoint rather than training from
   scratch. Sized for the available GPU; training from scratch is not.
3. Set `causal=True`, which implies `norm_type="cLN"`.
4. Watch for the expected quality drop. Causal models are worse than
   non-causal ones at the same size, always. The question is how much, measured
   the way the probe measures: how far the interferer's score falls against how
   far the target's falls.
5. Re-run `python -m pvi.probe` on `session2.wav` with the new checkpoint as a third
   backend in `pvi/tse/`, and compare against the numbers at the top of this
   file.

### Success criterion

The causal 16 kHz model separates at least as well as the non-causal 8 kHz one
does today: interferer's score falling at least 2x further than the target's,
and the residual leaning clearly toward the interferer.

### Risks

- Causal degradation may be severe enough to need a larger model, which fights
  Phase 1's latency budget. If so, the trade is short lookahead: a few tens of
  milliseconds of future context is within CLAUDE.md invariant 6's ~50 ms.
- LibriMix is read speech at equal loudness. Real calls are conversational at
  unequal loudness. If transfer degrades after retraining, DNS-Challenge data
  is the answer, not more LibriMix.

---

## Phase 3 — Decide what happens above 8 kHz

Even at 16 kHz the extractor band-limits its output to 8 kHz, while the audio
path is 48 kHz. Two options:

- **Accept it.** Wideband speech at 8 kHz bandwidth sounds genuinely good and
  is what most VoIP delivers anyway.
- **Band-split.** Run the mask on the low band and apply its gain envelope to
  the 8-24 kHz band, which mostly carries sibilance and brightness. Cheap, but
  a single broadband gain cannot separate two voices in the high band, so it
  helps most when one speaker dominates.

**Defer this until you have heard Phase 2's output.** It may simply be good
enough, and this is the easiest place in the plan to waste effort on a problem
that does not exist.

---

## Phase 4 — Integrate the mask with the gate

### The chain

```
mic 48 kHz
  -> DeepFilterNet3          removes noise and reverb        (0.30 ms/frame)
  -> resample 48k -> 16k
  -> mask estimator          ducks the interfering voice     (Phase 1 target)
  -> resample 16k -> 48k
  -> gate gain               mutes when you are not talking  (free, other thread)
  -> virtual device -> the call
```

### Why both

They solve different problems, measured: 1.1 dB of attenuation from the mask on
interferer-only audio against 34 dB from the gate. The mask cannot mute you
when you are silent because it never saw that case in training. The gate cannot
touch overlap because a gate decides when, not what.

The gate is also nearly free. ECAPA costs 13.5 ms every 200 ms on a background
thread, about 7% of one core, and contributes nothing to callback latency
because it only sets a gain target.

### Steps

1. Add the mask to the callback, behind a flag, defaulting off until it has
   run for a week without dropouts.
2. Keep the existing ramping. Gain changes are still ramped (invariant 3) and
   the closed-gate floor is still `FLOOR`, not zero (invariant 4).
3. Decide the ordering question by ear: mask-then-gate, or gate-then-mask.
   Mask-then-gate is the default here because the mask wants the fullest
   signal it can get.
4. Measure end-to-end latency for real, which nothing has done yet.

---

## Phase 5 — The product

### The menu bar app

Toggle on/off, pick your enrolled profile, see whether the gate is open. The
Python realtime loop is the engine; the UI is a thin shell over it.

### Creating the aggregate device

Users must not be asked to open Audio MIDI Setup. The app should create the
aggregate device containing their microphone and the virtual cable, and select
it. This is a CoreAudio call from a small Swift helper. It is the single
largest usability difference between "a script the author runs" and "a thing
other people install".

### Also required

- Bundle or install the virtual audio device (BlackHole on macOS).
- First-run enrollment as a guided UI rather than a terminal prompt.
- Ship the CC-BY-4.0 attribution for the model weights.
- Decide what happens to `speakers.npz`. It is biometric data. It should stay
  on the user's machine, and the app should say so plainly.

---

## Ordering, and why

1. **Phase 1**, because it is cheap and it decides whether the rest is
   possible.
2. **Phase 2**, the long pole, and the only phase needing a GPU.
3. **Phase 4**, integration, once there is something to integrate.
4. **Phase 3**, only if Phase 2's output sounds band-limited in practice.
5. **Phase 5**, last. A menu bar app around a filter that does not work yet is
   the most expensive possible way to discover it does not work.

## Standing rules

- Nothing gets a performance claim until it has been measured on real
  recordings from the target room. This has already caught two wrong
  conclusions.
- The probe is the arbiter. When something changes, run `python -m pvi.probe` on
  `session2.wav` and compare against the numbers at the top of this file.
- Enrollment and inference share one preprocessing path. Breaking that raises
  nothing and silently invalidates every threshold.
