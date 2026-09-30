"""CPU-only smoke test: imports, one UDA (alignment + AdaBN) prediction, one CFT step.

No dataset download is required: the data are random tensors.
Run with:  pytest -q tests
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_CHANS, N_TIMES, N_CLASSES = 8, 128, 2


def test_imports():
    import moabb_datasets  # noqa: F401
    import train_transfer  # noqa: F401
    import tta_wrapper  # noqa: F401
    import utils  # noqa: F401
    from models.builder import build_model  # noqa: F401


@pytest.mark.parametrize("model_name", ["EEGNetv4", "ShallowConvNet"])
def test_uda_predict_and_cft_step(model_name):
    from models.builder import build_model
    from tta_wrapper import TTAWrapper

    torch.manual_seed(0)
    device = torch.device("cpu")
    model = build_model(model_name, N_CHANS, N_TIMES, N_CLASSES, device, {})
    args = SimpleNamespace(
        use_tta=True,
        alignment_type="euclidean",
        alignment_cov_epsilon=1e-6,
        alignment_transform_epsilon=1e-7,
        alignment_ref_ema_beta=0.9,
        use_adabn=True,
        adabn_mode="train_mode",
        tta_buffer_length=4,
        finetune_mode="full",
    )
    wrapped = TTAWrapper(model, args, sr_hz=128.0).to(device)

    # --- online UDA: alignment reference + AdaBN buffer, one trial at a time ---
    wrapped.eval()
    for _ in range(6):
        trial = torch.randn(1, N_CHANS, N_TIMES)
        logits = wrapped.update_tta_statistics_and_predict(trial)
        assert logits.shape == (1, N_CLASSES)
        assert torch.isfinite(logits).all()

    # --- one continual-finetuning (CFT) step on a window of recent trials ---
    wrapped.train()
    x = torch.randn(16, N_CHANS, N_TIMES)
    y = torch.randint(0, N_CLASSES, (16,))
    params = [p for p in wrapped.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    before = [p.detach().clone() for p in params]
    loss = torch.nn.functional.cross_entropy(wrapped(x, is_finetuning_batch=True), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, params))
