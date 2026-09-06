"""
Opening the duplex audio stream, and dealing with aggregate device channels.

An Aggregate Device stacks its members' channels end to end, so on a typical
"microphone plus virtual cable" aggregate the cable is NOT channel 0. Getting
this wrong is silent: the mic reads fine and the output goes to the
microphone's own headphone jack, so the far end hears nothing and no error is
raised anywhere.
"""

import sounddevice as sd

from .. import dsp


def parse_channels(spec):
    """"2,3" -> (2, 3). Raises ValueError on anything else."""
    return tuple(int(c) for c in str(spec).split(","))


def open_stream(inp, out, in_channel, out_channels, callback):
    """Open the duplex stream, with a readable failure for the macOS case.

    A duplex sd.Stream spans one input device and one output device. When those
    are two different physical devices they run on two different clocks, and
    CoreAudio will not bridge them. The fix is an Aggregate Device, which makes
    the OS do the bridging before PortAudio ever sees it.

    PortAudio cannot be asked for channel 5 without opening channels 0 through
    5, so open enough to reach the highest one needed and let the callback zero
    the rest.
    """
    try:
        return sd.Stream(device=(inp, out), samplerate=dsp.SR,
                         blocksize=dsp.FRAME,
                         channels=(in_channel + 1, max(out_channels) + 1),
                         dtype="float32",
                         latency="low", callback=callback)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not open a duplex stream on devices ({inp}, {out}): {e}\n\n"
            f"On macOS this usually means the mic and the virtual cable are "
            f"separate devices on separate clocks. Open Audio MIDI Setup, "
            f"create an Aggregate Device containing both, and pass its index "
            f"for --in and --out. `python live.py --list-devices` will show it "
            f"once it exists."
        ) from e
