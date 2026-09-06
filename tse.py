"""
Target-speech-extraction backends for probe.py.

Two implementations of the same idea (TD-SpeakerBeam), from two different
groups, with very different licences. That difference is the whole reason this
module exists as a separate file.

    speakerbeam   BUTSpeechFIT/speakerbeam, the original release.
                  EVALUATION ONLY. The BUT/NTT agreement grants use "internally
                  for the purposes of testing, analyzing, and evaluating", and
                  it binds automatically on install whether or not anyone reads
                  it. Running the probe is squarely inside that grant. Routing
                  a phone call through it is not, and neither is shipping
                  anything built on the weights.

    espnet        ESPnet's own TD-SpeakerBeam, trained by Wangyou Zhang on
                  LibriMix and published on Hugging Face. Code is Apache-2.0,
                  weights are CC-BY-4.0. Both permit redistribution,
                  modification and commercial use, with attribution.

The point of having both is to answer one question: does the licence-clean
model perform like the one we validated? If it does, the licence blocker
disappears without changing architecture. If it does not, we know the cost of
staying clean before committing to it.

Neither model is vendored. Both are fetched at runtime into models/, which is
gitignored, so nothing here redistributes anything.

Both backends expose the same interface, so probe.py does not care which is in
use:

    model = tse.load("espnet")          # a torch Module
    y = model(mix, enroll)              # (B, T) in, (B, T) out, both at 8 kHz

ARCHITECTURE, side by side, read out of the two checkpoints:

                        speakerbeam       espnet
    filterbank          512 x 16 / 8      256 x 32 / 16
    TCN blocks          8 x 3             8 x 4
    bottleneck          128               256
    speaker embedding   128               256
    adaptation          multiply at 7     multiply at 7
    sample rate         8000              8000
    causal              no                no

Same family, different sizes. The espnet encoder has a longer window (32
samples against 16) and half as many filters, which is a coarser time
resolution for the same amount of compute.
"""

import pathlib
import subprocess
import sys
import urllib.request

import numpy as np
import torch

import dsp

# Both models are 8 kHz. Kept here rather than in probe.py because it is a
# property of the checkpoints, not of the experiment.
SR = 8000

MODELS_DIR = dsp.MODEL_DIR

# --- backend 1: the original BUT release, evaluation licence ---------------

SPEAKERBEAM_REPO = "https://github.com/BUTSpeechFIT/speakerbeam.git"
SPEAKERBEAM_DIR = MODELS_DIR / "speakerbeam"

# --- backend 2: ESPnet, Apache-2.0 code and CC-BY-4.0 weights --------------

ESPNET_DIR = MODELS_DIR / "espnet_tse"
ESPNET_RAW = "https://raw.githubusercontent.com/espnet/espnet/master"
ESPNET_HF = "espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw"
ESPNET_CKPT_PATH = "exp/enh_train_raw/99epoch.pth"

# Only the modules the extraction path actually touches. Fetching these rather
# than `pip install espnet` keeps the dependency footprint at one small package
# (torch-complex) instead of the whole toolkit, which matters because every
# dependency is a barrier for someone running this on their own machine.
ESPNET_FILES = [
    "espnet2/enh/encoder/abs_encoder.py",
    "espnet2/enh/encoder/conv_encoder.py",
    "espnet2/enh/decoder/abs_decoder.py",
    "espnet2/enh/decoder/conv_decoder.py",
    "espnet2/enh/extractor/abs_extractor.py",
    "espnet2/enh/extractor/td_speakerbeam_extractor.py",
    "espnet2/enh/layers/adapt_layers.py",
    "espnet2/enh/layers/complex_utils.py",
    "espnet2/enh/layers/tcn.py",
    "espnet2/legacy/nets/pytorch_backend/nets_utils.py",
]

# Read out of the checkpoint's own config.yaml. Hardcoded rather than parsed so
# the probe does not need a YAML dependency for eleven numbers.
ESPNET_ENCODER = dict(channel=256, kernel_size=32, stride=16)
ESPNET_EXTRACTOR = dict(
    layer=8, stack=4, bottleneck_dim=256, hidden_dim=512, skip_dim=256,
    kernel=3, causal=False, norm_type="gLN", pre_nonlinear="prelu",
    nonlinear="relu", i_adapt_layer=7, adapt_layer_type="mul",
    adapt_enroll_dim=256, use_spk_emb=False,
)


def _fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"  fetching {dest.name}")
    urllib.request.urlretrieve(url, dest)
    return dest


# --- speakerbeam ----------------------------------------------------------

def ensure_speakerbeam(root=SPEAKERBEAM_DIR):
    """Clone TD-SpeakerBeam and put its src/ on sys.path.

    Cloned at runtime rather than vendored, for the licence reason at the top
    of this module. The upstream modules import each other as
    `from models.base_models_informed import ...`, with no package prefix, so
    src/ itself has to be the sys.path entry.
    """
    root = pathlib.Path(root)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning TD-SpeakerBeam into {root} (about 30 MB)")
        subprocess.run(["git", "clone", "--depth", "1", SPEAKERBEAM_REPO, str(root)],
                       check=True)
    ckpt = root / "example" / "model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} is missing; delete {root} and re-run")
    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return ckpt


def load_speakerbeam(device="cpu"):
    """Build TimeDomainSpeakerBeam and load the published weights.

    Deliberately does NOT use asteroid's `Model.from_pretrained`, which the
    upstream demo notebook calls. It passes no `weights_only` to torch.load,
    which since torch 2.6 defaults to True and fails, and it routes through
    `huggingface_hub.cached_download`, which newer releases removed. Both are
    first-run crashes on a current environment.
    """
    ckpt = ensure_speakerbeam()
    from models.td_speakerbeam import TimeDomainSpeakerBeam

    try:
        conf = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"note: safe load failed ({type(e).__name__}), retrying unrestricted")
        conf = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model_args", "state_dict"):
        if key not in conf:
            raise ValueError(f"checkpoint has no '{key}'; got {list(conf)}")
    model = TimeDomainSpeakerBeam(**conf["model_args"])
    model.load_state_dict(conf["state_dict"])
    return model.eval().to(device)


# --- espnet ---------------------------------------------------------------

def ensure_espnet(root=ESPNET_DIR):
    """Fetch the ESPnet modules and checkpoint, and make them importable.

    The empty __init__.py files are deliberate. Importing the real
    espnet2/__init__.py would pull in the rest of the toolkit; these stubs make
    `espnet2.enh.extractor...` resolve to exactly the ten files we fetched.
    Inserted at the front of sys.path so this wins even if espnet is installed.
    """
    root = pathlib.Path(root)
    for rel in ESPNET_FILES:
        _fetch(f"{ESPNET_RAW}/{rel}", root / rel)
    # Package stubs for every directory on the path to those modules.
    pkgs = set()
    for rel in ESPNET_FILES:
        parts = pathlib.Path(rel).parts[:-1]
        for i in range(1, len(parts) + 1):
            pkgs.add(pathlib.Path(*parts[:i]))
    for pkg in pkgs:
        init = root / pkg / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        if not init.exists():
            init.write_text("")

    ckpt = root / "ckpt" / "99epoch.pth"
    if not ckpt.exists():
        print(f"fetching ESPnet TD-SpeakerBeam weights (about 65 MB)")
        _fetch(f"https://huggingface.co/{ESPNET_HF}/resolve/main/{ESPNET_CKPT_PATH}",
               ckpt)

    p = str(root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    return ckpt


class EspnetTSE(torch.nn.Module):
    """ESPnet's encoder / extractor / decoder wired up as one callable.

    ESPnet normally reaches this through its own inference machinery, which
    wants the full toolkit and a task config. The checkpoint is a plain state
    dict of three submodules, so wiring them directly is both smaller and
    easier to reason about. The plumbing mirrors ESPnetExtractionModel's
    forward_enhance, with share_encoder=True as the config specifies: the
    enrollment goes through the SAME encoder as the mixture.
    """

    def __init__(self, encoder, extractor, decoder):
        super().__init__()
        self.encoder, self.extractor, self.decoder = encoder, extractor, decoder

    def forward(self, mix, enroll):
        """(B, T) mixture and (B, T') enrollment in, (B, T) estimate out."""
        il = torch.full((mix.shape[0],), mix.shape[1], dtype=torch.long,
                        device=mix.device)
        ia = torch.full((enroll.shape[0],), enroll.shape[1], dtype=torch.long,
                        device=enroll.device)
        f_mix, flens = self.encoder(mix, il)
        f_aux, flens_aux = self.encoder(enroll, ia)
        f_pre, _, _ = self.extractor(f_mix, flens, f_aux, flens_aux)
        out, _ = self.decoder(f_pre, il)
        return out


def load_espnet(device="cpu"):
    """Build the ESPnet TD-SpeakerBeam and load the CC-BY-4.0 weights."""
    ckpt = ensure_espnet()
    from espnet2.enh.decoder.conv_decoder import ConvDecoder
    from espnet2.enh.encoder.conv_encoder import ConvEncoder
    from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor

    encoder = ConvEncoder(**ESPNET_ENCODER)
    decoder = ConvDecoder(**ESPNET_ENCODER)
    extractor = TDSpeakerBeamExtractor(input_dim=ESPNET_ENCODER["channel"],
                                       **ESPNET_EXTRACTOR)
    model = EspnetTSE(encoder, extractor, decoder)

    # No pickle globals in this checkpoint at all, so the safe loader works.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing weights for: {missing[:5]}")
    if unexpected:
        print(f"note: ignoring {len(unexpected)} keys not used at inference")
    return model.eval().to(device)


# --- registry -------------------------------------------------------------

BACKENDS = {
    "speakerbeam": (
        load_speakerbeam,
        "BUT/NTT evaluation-only. Fine for this probe, cannot be shipped.",
    ),
    "espnet": (
        load_espnet,
        "Apache-2.0 code, CC-BY-4.0 weights. Redistributable with attribution.",
    ),
}


def load(name, device="cpu"):
    """Load a backend by name. Returns a torch Module taking (mix, enroll)."""
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; pick from {list(BACKENDS)}")
    fn, licence = BACKENDS[name]
    print(f"backend: {name}  [{licence}]")
    return fn(device=device)
