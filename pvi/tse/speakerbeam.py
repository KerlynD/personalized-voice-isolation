"""
Backend: BUTSpeechFIT/speakerbeam, the original TD-SpeakerBeam release.

LICENCE: EVALUATION ONLY. NOT SHIPPABLE.
----------------------------------------
The BUT/NTT agreement grants a licence to use the software "internally for the
purposes of testing, analyzing, and evaluating the methods or mechanisms as
shown in the research paper", and clause 3 makes it bind automatically "upon
User's installing, accessing, and using the Software, even if User has not
expressly accepted" it.

Running the probe is squarely inside that grant: the probe is evaluation.
Routing a phone call through these weights is not, and neither is shipping
anything derived from them, and neither is inviting other people to install it
for daily use, since they would each accept an evaluation licence and then
exceed it.

This backend exists only as the reference the project's first extraction result
was measured against. Build on pvi.tse.espnet.

Nothing is vendored. The repo is cloned at runtime into models/, which is
gitignored, so this file redistributes nothing.
"""

import pathlib
import subprocess
import sys

import torch

from ..dsp import MODEL_DIR

REPO = "https://github.com/BUTSpeechFIT/speakerbeam.git"
DIR = MODEL_DIR / "speakerbeam"

LICENCE = "BUT/NTT evaluation-only. Fine for the probe, cannot be shipped."


def ensure(root=DIR):
    """Clone TD-SpeakerBeam and put its src/ on sys.path.

    The upstream modules import each other as `from models.base_models_informed
    import ...`, with no package prefix, so src/ itself has to be the sys.path
    entry rather than its parent.
    """
    root = pathlib.Path(root)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning TD-SpeakerBeam into {root} (about 30 MB)")
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(root)],
                       check=True)
    ckpt = root / "example" / "model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} is missing; delete {root} and re-run")
    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return ckpt


def load(device="cpu"):
    """Build TimeDomainSpeakerBeam and load the published weights.

    Deliberately does NOT use asteroid's `Model.from_pretrained`, which the
    upstream demo notebook calls. Two first-run crashes on a current
    environment: it passes no `weights_only` to torch.load, which has defaulted
    to True since torch 2.6, and it routes through
    `huggingface_hub.cached_download`, which newer releases removed outright.

    Doing the two steps by hand skips both. weights_only=True is tried first
    and is expected to succeed, since this checkpoint's pickle references only
    OrderedDict, torch.FloatStorage and torch._utils._rebuild_tensor_v2, all of
    which are on torch's allowlist.
    """
    ckpt = ensure()
    from models.td_speakerbeam import TimeDomainSpeakerBeam

    try:
        conf = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"note: safe load failed ({type(e).__name__}), retrying unrestricted")
        conf = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model_args", "state_dict"):
        if key not in conf:
            raise ValueError(f"checkpoint has no '{key}'; got {list(conf)}")
    model = TimeDomainSpeakerBeam(**conf["model_args"])
    model.load_state_dict(conf["state_dict"])
    return model.eval().to(device)
