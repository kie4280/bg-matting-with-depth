# This is the custom code of V2 with depth info


import cv2
import numpy as np

import torch
from torch import tensor
from MiDas import utils
import PIL

from torchvision.transforms import Compose
from MiDas.midas.dpt_depth import DPTDepthModel
from MiDas.midas.transforms import Resize, NormalizeImage, PrepareForNet


class Midas_depth:
    def __init__(self, model_path="/eva_data/kie/research/pretrained/intel-MiDas-dpt-large.pt",
                 device: str = "cuda:1", optimize=True) -> None:

        print("initialize MiDas")
        self.optimize = optimize
        # select device
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")

        print("device: {}".format(self.device))

        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.model.eval()

        if optimize == True:
            # rand_example = torch.rand(1, 3, net_h, net_w)
            # model(rand_example)
            # traced_script_module = torch.jit.trace(model, rand_example)
            # model = traced_script_module

            if self.device == torch.device("cuda"):
                self.model = self.model.to(memory_format=torch.channels_last)
                self.model = self.model.half()

        self.model.to(self.device)

    def inference(self, imgs: np.ndarray) -> torch.Tensor:
        predictions = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_input = self.transform({"image": img})["image"]

            # compute
            with torch.no_grad():
                sample = torch.from_numpy(img_input).to(
                    self.device).unsqueeze(0)
                if self.optimize == True and self.device == torch.device("cuda"):
                    sample = sample.to(memory_format=torch.channels_last)
                    sample = sample.half()
                prediction = self.model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )

                )
                predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)

        return predictions


def Normalize(depth_tensor: torch.Tensor) -> torch.Tensor:
    max_d = depth_tensor.max()
    min_d = depth_tensor.min()
    norm = (depth_tensor - min_d) / (max_d - min_d)
    return norm


if __name__ == "__main__":
    MD = Midas_depth()
    img = [(cv2.imread(
        "/eva_data/kie/data/training/background/custom/lib_3-98.png") / 255.0)]
    output = MD.inference(img).squeeze_(dim=0)

    white = 255 * torch.ones([3, output.shape[1], output.shape[2]],
                       dtype=torch.float32, device='cuda:1')
    output = Normalize(output) * white
    output = output.moveaxis(0, -1)
    cv2.imwrite("test.png", output.to(torch.uint8).cpu().numpy())
