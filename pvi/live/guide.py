"""
The scripted recording session, for producing evaluation audio.

Recording a probe session used to mean watching a clock that was not on screen
and remembering a four-phase script. This prints where you are, rings the
terminal bell at every handover because you will be talking rather than reading
the screen, and stops on its own.
"""

import time

# (seconds, what to do). Only the BOTH phase is the experiment. The
# single-speaker phases are controls: they let you hear what the gate does
# correctly, so that when you listen to the overlap you are judging the
# extraction and not re-judging the gate. Sixty seconds of genuine overlap
# yields several usable stretches; going longer mostly adds probe runtime,
# since the model is far slower than real time.
PROBE_SCRIPT = [
    (20.0, "YOU alone"),
    (20.0, "THEM alone"),
    (60.0, "BOTH at once, talk over each other"),
    (20.0, "YOU alone"),
]


def mmss(t):
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def phases(script):
    """(start, end, label) for each phase, in seconds from the script start."""
    out, t = [], 0.0
    for dur, label in script:
        out.append((t, t + dur, label))
        t += dur
    return out


def run(pipeline, script=PROBE_SCRIPT):
    """Walk the operator through a timed session. Returns the lead-in seconds.

    Everything prints from here rather than from the analysis thread, because
    both would be writing the same terminal line with a carriage return and the
    result is unreadable. Pipeline.monitor is forced off when this runs.

    The returned lead-in matters: the stream is already recording while the
    plan prints and the count-in runs, so every phase sits that far into the
    file, and the probe command printed afterwards has to account for it.
    """
    t_entry = time.time()
    ph = phases(script)
    total = ph[-1][1]

    print("\nSession plan:")
    for a, b, label in ph:
        print(f"    {mmss(a)} - {mmss(b)}   {label}")
    print(f"\n  Total {mmss(total)}. It stops on its own. Ctrl+C stops early and")
    print("  the recording so far is still usable.")
    for i in (3, 2, 1):
        print(f"  starting in {i}...", flush=True)
        time.sleep(1.0)

    start = time.time()
    lead_in = start - t_entry
    current = -1
    try:
        while True:
            el = time.time() - start
            if el >= total:
                break
            i = next(j for j, (_, b, _) in enumerate(ph) if el < b)
            if i != current:
                current = i
                print(f"\n\a>>> {mmss(ph[i][0])}  {ph[i][2]}", flush=True)
            left = ph[i][1] - el
            bar = "#" * int(max(0.0, pipeline.score) * 30)
            print(f"\r  {mmss(el)}/{mmss(total)}  {left:4.0f}s left   "
                  f"sim={pipeline.score:+.3f} gain={pipeline.gain:.2f}  "
                  f"{bar:<30}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  stopped early")
        return lead_in
    print("\n\a  done")
    return lead_in
