"""
Command line arguments example:

    python inference_webcam.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --resolution 1280 720

"""

import argparse
import time
import cv2
import torch

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import numpy as np

from MiDaS import utils

from torchvision.transforms import Compose
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

from V2.model import MattingBase, MattingRefine

filename = "fast_moving"
video_format = "MOV"
test_data_path = "/home/kie/personal_data"

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference from video')

parser.add_argument('--V2-model-type', type=str, default="mattingrefine",
                    choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--V2-model-backbone', type=str, default="resnet50",
                    choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--V2-model-backbone-scale', type=float, default=0.25)
parser.add_argument('--V2-model-checkpoint', type=str, required=True)

parser.add_argument('--V2-model-refine-mode', type=str,
                    default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--V2-model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--V2-model-refine-threshold', type=float, default=0.7)



parser.add_argument("--Midas-model-checkpoint", type=str, required=True)
parser.add_argument("--Midas-model-type", type=str, default="large")

parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2,
                    metavar=('width', 'height'), default=(1920, 1080))

args = parser.parse_args()

# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id="/home/kie/personal_data/{}.{}".format(filename, video_format)):
        self.capture = cv2.VideoCapture(device_id)
        if (not self.capture.isOpened()):
            print("cannot open input video")
            exit()
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # self.success_reading, self.frame = self.capture.read()

    def read(self):
        success, frame = self.capture.read()
        if (not success):
            return False, None
        return success, frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()

# An FPS tracker that computes exponentialy moving average FPS


class FPSTracker:
    def __init__(self, ratio=0.5):
        self._last_tick = None
        self._avg_fps = None
        self.ratio = ratio

    def tick(self):
        if self._last_tick is None:
            self._last_tick = time.time()
            return None
        t_new = time.time()
        fps_sample = 1.0 / (t_new - self._last_tick)
        self._avg_fps = self.ratio * fps_sample + \
            (1 - self.ratio) * \
            self._avg_fps if self._avg_fps is not None else fps_sample
        self._last_tick = t_new
        return self.get()

    def get(self):
        return self._avg_fps

# Wrapper for playing a stream with cv2.imshow(). It can accept an image and return keypress info for basic interactivity.
# It also tracks FPS and optionally overlays info onto the stream.


class Displayer:
    def __init__(self, title, width, height, frame_rate, show_info=True):
        self.title, self.width, self.height = "out_" + title, width, height
        self.show_info = show_info
        self.fps_tracker = FPSTracker()
        self.videowriter = cv2.VideoWriter(self.title + '.avi', cv2.VideoWriter_fourcc(
            *'MPEG'), frame_rate, (self.width, self.height))

    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
        self.videowriter.write(image)

    def __exit__(self, exec_type, exc_value, traceback):
        self.videowriter.release()


class BGv2:
    # Load model

    def __init__(self, args):
        self.args = args
        if args.V2_model_type == 'mattingbase':
            self.model = MattingBase(args.V2_model_backbone)
        if args.V2_model_type == 'mattingrefine':
            self.model = MattingRefine(
                args.V2_model_backbone,
                args.V2_model_backbone_scale,
                args.V2_model_refine_mode,
                args.V2_model_refine_sample_pixels,
                args.V2_model_refine_threshold)

        self.model = self.model.cuda().eval()
        self.model.load_state_dict(torch.load(
            args.V2_model_checkpoint), strict=False)
        self.__background = None

    def set_bg(self, bgr):
        self.__background = cv2_frame_to_cuda(bgr)

    def single_frame(self, frame):
        with torch.no_grad():
            if self.__background is not None:  # matting
                src = cv2_frame_to_cuda(frame)
                pha, fgr = self.model(src, self.__background)[:2]
                res = pha * fgr + (1 - pha) * torch.ones_like(fgr)
                res = res.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()[0]
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
                return res

        return None


class MidasSingle:

    def __init__(self, args, optimize=True) -> None:
        """Run MonoDepthNN to compute depth maps.

        Args:
            model_path (str): path to saved model
            optimize (bool): 
        """

        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.optimize = optimize
        # select device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if args.Midas_model_type == "large":
            self.model = MidasNet(
                args.Midas_model_checkpoint, non_negative=True)
            net_w, net_h = 384, 384
        elif args.Midas_model_type == "small":
            self.model = MidasNet_small(args.Midas_model_checkpoint, features=64, backbone="efficientnet_lite3",
                                        exportable=True, non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
        else:
            print(
                f"model_type '{args.Midas_model_type}' not implemented, use: --model_type large")
            assert False

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

        self.model.eval()

        if optimize == True:
            rand_example = torch.rand(1, 3, net_h, net_w)
            self.model(rand_example)
            traced_script_module = torch.jit.trace(self.model, rand_example)
            self.model = traced_script_module

            if self.device == torch.device("cuda"):
                self.model = self.model.to(memory_format=torch.channels_last)
                self.model = self.model.half()

        self.model.to(self.device)

    def single_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        img_input = self.transform({"image": img})["image"]
        # compute
        with torch.no_grad():
            time1 = time.time()
            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
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
                .squeeze()
                .cpu()
                .numpy()
            )
            
            
        # output
        
        output = self._write_depth(prediction, bits=1)
        time1 = time.time() - time1
        # print(1.0/time1)

        return output

    def _write_depth(self, depth, bits=1):
        """Write depth map to pfm and png file.

        Args:
            path (str): filepath without extension
            depth (array): depth
        """
        # write_pfm(path + ".pfm", depth.astype(np.float32))
        time1 = time.time()
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.type)

        if bits == 1:
            time1 = time.time() - time1
            print(time1)
            return out.astype("uint8")
        elif bits == 2:
            return out.astype("uint16")

        raise RuntimeError("Invalid bits")

# ---------- Utility functions -----------


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(frame).unsqueeze_(0).cuda()


def depth_mask(frame_orig, frame_depth, bgr, threshold: int = 80):
    frame1 = np.where(frame_depth < threshold, 1, 0)
    frame = np.zeros(frame_orig.shape, dtype='uint8')
    for i in range(3):
        frame[:, :, i] = frame1 * frame_orig[:, :, i] + \
            (1-frame1) * bgr[:, :, i]
    return frame


# --------------- Main ---------------
if __name__ == "__main__":

    cam = Camera()
    dsp = Displayer(filename, cam.width, cam.height,
                    cam.fps, show_info=(not args.hide_fps))
    bgr = cv2.imread("/home/kie/personal_data/{}_bgr.png".format(filename))
    if (cam.width < cam.height):
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    v = BGv2(args)
    v.set_bg(bgr)
    mi = MidasSingle(args)

    time1 = 0
    while True:
        has_next, frame = cam.read()
        if has_next is False:
            break

        time1 = time.time()
        frame_depth = mi.single_frame(frame)
        # frame = depth_mask(frame, frame_depth, bgr, 70)
        time1 = time.time() - time1
        # frame = v.single_frame(frame)

        # dsp.step(frame)
        # print("fps:", 1.0/time1)
