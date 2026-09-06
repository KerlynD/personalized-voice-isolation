"""
Backend: ESPnet's TD-SpeakerBeam. THIS IS THE ONE TO BUILD ON.

LICENCE: Apache-2.0 code, CC-BY-4.0 weights.
-------------------------------------------
Trained by Wangyou Zhang on LibriMix and published on Hugging Face. Both
licences permit redistribution, modification and commercial use with
attribution, which is what makes a shippable product possible at all.

Measured against the evaluation-only BUT release on the same recording, it is
the better model rather than a compromise:

                        espnet (CC-BY)   speakerbeam (eval only)
    their score fell         0.049            0.041
    your score fell          0.021            0.038
    ratio                    2.3x             1.1x

It also trains with plain SNR rather than SI-SNR, so its output is not
scale-invariant and arrives at a sane level. The BUT checkpoint overshoots by
roughly 350,000x and needs pvi.probe.audio.match_scale to be audible at all.

WHAT IS STILL WRONG WITH IT
---------------------------
Non-causal in two ways: global layer norm normalizes across the whole
utterance, and the dilated convolutions look ahead. And 8 kHz, so the output is
band-limited to 4 kHz. Both need retraining to fix; see docs/PLAN.md phase 2.
The `causal` flag below is the architecture's, and these weights were not
trained with it set.

Nothing is vendored. Only the ten modules the extraction path actually touches
are fetched at runtime into gitignored models/, which keeps this to one added
dependency (torch-complex) rather than the whole ESPnet toolkit.
"""

import pathlib
import sys
import urllib.request

import torch

from ..dsp import MODEL_DIR

DIR = MODEL_DIR / "espnet_tse"
RAW = "https://raw.githubusercontent.com/espnet/espnet/master"
HF_REPO = "espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw"
HF_CKPT = "exp/enh_train_raw/99epoch.pth"

LICENCE = "Apache-2.0 code, CC-BY-4.0 weights. Redistributable with attribution."

FILES = [
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
# the package does not need a YAML dependency for eleven numbers.
ENCODER_CONF = dict(channel=256, kernel_size=32, stride=16)
EXTRACTOR_CONF = dict(
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


def ensure(root=DIR):
    """Fetch the ESPnet modules and checkpoint, and make them importable.

    The empty __init__.py stubs are deliberate. Importing the real
    espnet2/__init__.py would pull in the rest of the toolkit; these make
    `espnet2.enh.extractor...` resolve to exactly the files we fetched.
    Inserted at the front of sys.path so this wins even where espnet is
    installed for other reasons.
    """
    root = pathlib.Path(root)
    for rel in FILES:
        _fetch(f"{RAW}/{rel}", root / rel)

    pkgs = set()
    for rel in FILES:
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
        print("fetching ESPnet TD-SpeakerBeam weights (about 65 MB)")
        _fetch(f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_CKPT}", ckpt)

    p = str(root.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    return ckpt


class EspnetTSE(torch.nn.Module):
    """ESPnet's encoder, extractor and decoder wired up as one callable.

    ESPnet normally reaches these through its own inference machinery, which
    wants the full toolkit and a task config. The checkpoint is a plain state
    dict of three submodules, so wiring them directly is both smaller and
    easier to reason about. The plumbing mirrors ESPnetExtractionModel's
    forward_enhance with share_encoder=True as the config specifies: the
    enrollment goes through the SAME encoder as the mixture.

    Speed note for phase 1 of docs/PLAN.md. One forward costs about 24 ms
    almost regardless of how much audio it is given, because it is 130 Conv1d
    layers at roughly 190 us of PyTorch dispatch each. The arithmetic is
    negligible; the framework is the cost. Splitting the enrollment out with
    embed_enrollment() removes a further 37 ms per 3 s of reference audio,
    which is pure waste when the embedding never changes during a call.
    """

    def __init__(self, encoder, extractor, decoder):
        super().__init__()
        self.encoder, self.extractor, self.decoder = encoder, extractor, decoder

    def embed_enrollment(self, enroll):
        """(B, T) enrollment -> the fixed speaker embedding the masker wants.

        Depends only on the enrollment, so it is computed once and reused for
        every chunk of a call. Recomputing it per chunk is what made the probe
        slow.
        """
        ia = torch.full((enroll.shape[0],), enroll.shape[1], dtype=torch.long,
                        device=enroll.device)
        f_aux, _ = self.encoder(enroll, ia)
        return self.extractor.auxiliary_net(
            f_aux.transpose(1, 2)).squeeze(1).mean(dim=-1)

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


def load(device="cpu"):
    """Build the ESPnet TD-SpeakerBeam and load the CC-BY-4.0 weights."""
    ckpt = ensure()
    from espnet2.enh.decoder.conv_decoder import ConvDecoder
    from espnet2.enh.encoder.conv_encoder import ConvEncoder
    from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor

    model = EspnetTSE(
        ConvEncoder(**ENCODER_CONF),
        TDSpeakerBeamExtractor(input_dim=ENCODER_CONF["channel"],
                               **EXTRACTOR_CONF),
        ConvDecoder(**ENCODER_CONF),
    )
    # This checkpoint has no pickle globals at all, so the safe loader works.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing weights for: {missing[:5]}")
    if unexpected:
        print(f"note: ignoring {len(unexpected)} keys not used at inference")
    return model.eval().to(device)
