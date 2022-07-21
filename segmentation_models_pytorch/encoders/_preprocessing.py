import numpy as np
import torchvision.transforms.functional as TF

def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        x = TF.normalize(x, mean=np.array(mean), std=np.array(std))

    return x
