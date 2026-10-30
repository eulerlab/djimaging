import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")


def _stack(nx=16, ny=12, nt=10, seed=0):
    return np.random.default_rng(seed).random((nx, ny, nt)) * 100


@pytest.mark.parametrize("in_channels", [1, 2])
def test_unet_create_mask_from_data_shape(in_channels):
    from djimaging.autorois.unet import UNet

    model = UNet(in_channels=in_channels, channels=[4, 8]).eval()
    ch0, ch1 = _stack(seed=0), _stack(seed=1)

    with torch.no_grad():
        mask = model.create_mask_from_data(ch0_stack=ch0, ch1_stack=ch1, n_artifact=1, pixel_size_um=(2., 2.))

    assert mask.shape == ch0.shape[:2]


def test_unet_single_channel_does_not_need_ch1():
    from djimaging.autorois.unet import UNet

    model = UNet(in_channels=1, channels=[4, 8]).eval()
    with torch.no_grad():
        mask = model.create_mask_from_data(ch0_stack=_stack())
    assert mask.shape == (16, 12)


def test_unet_two_channels_require_ch1():
    from djimaging.autorois.unet import UNet

    model = UNet(in_channels=2, channels=[4, 8]).eval()
    with pytest.raises(ValueError):
        model.create_mask_from_data(ch0_stack=_stack())


def test_unet_multiple_stacks_are_concatenated():
    from djimaging.autorois.unet import UNet

    model = UNet(in_channels=1, channels=[4, 8]).eval()
    with torch.no_grad():
        mask = model.create_mask_from_data(ch0_stack=[_stack(seed=0), _stack(seed=1)], multiple_stacks=True)
    assert mask.shape == (16, 12)


def _install_fake_mediar(monkeypatch):
    """Fake ``mediar`` package, so that the wrapper can be tested without the real model."""

    class FakeFormer(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.p = torch.nn.Parameter(torch.zeros(1))

    class FakePredictor:
        def __init__(self, model, device, input_path, output_path, algo_params):
            self.algo_params = algo_params

        def _inference(self, img):
            assert img.shape[1] == 3
            return torch.zeros((img.shape[0], 3, *img.shape[2:]))

        def _post_process(self, pred):
            mask = np.zeros(pred.shape[1:], dtype=np.int32)
            mask[:100, :100] = 1
            mask[300:, 300:] = 5  # non-consecutive on purpose
            return mask

    mods = {
        "mediar": types.ModuleType("mediar"),
        "mediar.core": types.ModuleType("mediar.core"),
        "mediar.core.MEDIAR": types.ModuleType("mediar.core.MEDIAR"),
        "mediar.train_tools": types.ModuleType("mediar.train_tools"),
        "mediar.train_tools.models": types.ModuleType("mediar.train_tools.models"),
    }
    mods["mediar.core.MEDIAR"].Predictor = FakePredictor
    mods["mediar.train_tools.models"].MEDIARFormer = FakeFormer
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_mediar_wrapper_with_fake_model(monkeypatch, tmp_path):
    pytest.importorskip("torchvision")
    _install_fake_mediar(monkeypatch)
    from djimaging.autorois.mediar_wrapper import MediarWrapper

    path = tmp_path / "weights.pth"
    torch.save({"p": torch.zeros(1)}, path)

    wrapper = MediarWrapper(model_path=str(path))
    ch0 = _stack(nx=64, ny=64)
    mask = wrapper.create_mask_from_data(ch0_stack=ch0, n_artifact=1, pixel_size_um=(2., 2.))

    assert mask.shape == (64, 64)
    assert sorted(np.unique(mask)) == [0, 1, 2]  # consecutive IDs


def test_mediar_stack_to_image_is_normalized():
    from djimaging.autorois.mediar_wrapper import MediarWrapper

    img = MediarWrapper.stack_to_image(_stack())
    assert img.shape == (16, 12)
    assert img.min() == 0 and img.max() == pytest.approx(1.0)
