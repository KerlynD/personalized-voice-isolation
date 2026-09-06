"""
Personalized voice isolation: a microphone filter that passes one enrolled
speaker and suppresses everything else.

    pvi.dsp      shared signal path. Every other package imports it and none
                 of them reimplements any part of it, because enrollment and
                 inference drifting apart is this project's classic silent bug.
    pvi.enroll   records reference clips, builds the voiceprint and threshold
    pvi.live     the realtime loop: mic -> denoise -> gate -> virtual output
    pvi.tse      target-speech-extraction backends, and their licences
    pvi.probe    offline extraction test on real recordings
    pvi.bench    timing against the audio callback budget

Entry points are the four scripts at the repository root, which are thin
wrappers over pvi.<package>.cli. See docs/PLAN.md for where this is going and
.claude/CLAUDE.md for the invariants that must not be broken.
"""

__version__ = "0.2.0"
