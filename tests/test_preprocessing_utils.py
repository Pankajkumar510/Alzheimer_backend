import numpy as np

from src.preprocessing.utils import volume_to_multislice_rgb, resize_image


def test_multislice_and_resize():
    vol = np.random.rand(10, 64, 64).astype(np.float32)
    rgb = volume_to_multislice_rgb(vol, axis=0)
    assert rgb.shape[-1] == 3
    img = resize_image(rgb, (224, 224))
    assert img.shape == (224, 224, 3)
