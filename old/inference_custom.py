"""
Command line arguments example:

    python inference_custom.py \
        --V2-model-checkpoint \
            /eva_data/kie//research/pretrained/V2-model.pth \
        --Midas-model-checkpoint \
            /eva_data/kie/research/pretrained/intel-MiDas-model.pt \
        --Midas-model-type large \
        --file-list "old/input_file1.txt"

"""

import argparse
import time
from typing import Tuple
import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from V2.model import MattingBase, MattingRefine
from tqdm import tqdm

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
parser.add_argument('--V2-model-refine-sample-pixels',
                    type=int, default=80_000)
parser.add_argument('--V2-model-refine-threshold', type=float, default=0.7)


parser.add_argument("--Midas-model-checkpoint", type=str, required=True)
parser.add_argument("--Midas-model-type", type=str, default="large")

parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2,
                    metavar=('width', 'height'), default=(1920, 1080))
parser.add_argument("--file-list", type=str, required=True)

args = parser.parse_args()

# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id):
        self.capture = cv2.VideoCapture(device_id)
        if (not self.capture.isOpened()):
            print("cannot open input video")
            exit()
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.vertical = self.height > self.width
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Tuple[bool, np.ndarray]:
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


class Displayer:
    def __init__(self, title, width, height, frame_rate, show_info=True, msg=None):
        self.title, self.width, self.height = title, width, height
        self.show_info = show_info
        self.msg = msg
        self.fps_tracker = FPSTracker()
        self.videowriter = cv2.VideoWriter(self.title + '.avi', cv2.VideoWriter_fourcc(
            *'MPEG'), frame_rate, (self.width, self.height))

    # Update the currently showing frame and return key press char code
    def step(self, image):
        fps_estimate = self.fps_tracker.tick()
        if self.show_info and fps_estimate is not None:
            message = f"{int(fps_estimate)} fps | {self.width}x{self.height}"
            cv2.putText(image, message, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
        if self.msg is not None:
            cv2.putText(image, self.msg, (10, self.height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)
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
                # the return here is modified
                return res, pha

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

    def single_frame(self, frame, threshold=60) -> torch.Tensor:
        """
        compute the depth and output unnormalized depth map
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        img_input = self.transform({"image": img})["image"]

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        # compute
        with torch.no_grad():

            sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
            if self.optimize == True and self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = self.model.forward(sample)
            # end.record()
            # torch.cuda.synchronize()
            # print(start.elapsed_time(end))
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
            )
            max_T = torch.max(prediction)
            min_T = torch.min(prediction)
            prediction = 255.0 / (max_T - min_T) * (prediction - min_T)

        return prediction.to(dtype=torch.uint8)


# ---------- Utility functions -----------


def cv2_frame_to_cuda(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ToTensor()(frame).unsqueeze_(0).cuda()


def apply_mask(frame_orig: np.ndarray, frame_depth: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    frame = np.zeros(frame_orig.shape, dtype='uint8')
    for i in range(3):
        frame[:, :, i] = frame_depth * frame_orig[:, :, i] + \
            (1-frame_depth) * bgr[:, :, i]
    return frame


def filter_depth(frame_depth: torch.Tensor, threshold: int = 60) -> np.ndarray:
    with torch.no_grad():
        mask = torch.where(frame_depth > threshold, 1, 0)
        return mask.to(dtype=torch.uint8).cpu().numpy()

# --------------- Main ---------------


def process_full(filename, video_format="MOV"):
    cam = Camera(
        "/home/kie/personal_data/{}.{}".format(filename, video_format))
    dsp = Displayer("output/"+filename+"_full", cam.width, cam.height,
                    cam.fps, show_info=(not args.hide_fps), msg="Filters all")
    bgr = cv2.imread("/home/kie/personal_data/{}_bgr.png".format(filename))

    if (cam.vertical):
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    v2 = BGv2(args)
    v2.set_bg(bgr)
    mi = MidasSingle(args)
    blank = np.zeros((cam.height, cam.width, 3), dtype='uint8')
    blank.fill(255)
    tq = tqdm(total=cam.frame_count)
    dilate_kernel = np.ones((5, 5), dtype='uint8')

    while True:
        has_next, frame = cam.read()
        if has_next is False:
            break

        time1 = time.time()
        frame_depth = mi.single_frame(frame)
        frame_depth_front = filter_depth(frame_depth, 60)
        frame_depth_end = filter_depth(frame_depth, 80)
        frame_depth_front = cv2.dilate(
            frame_depth_front, dilate_kernel, iterations=5)
        frame_depth_end = cv2.dilate(
            frame_depth_end, dilate_kernel, iterations=5)
        frame_fused = apply_mask(frame, frame_depth_front, bgr)
        frame_matted = v2.single_frame(frame_fused)

        frame_matted = apply_mask(frame_matted, frame_depth_end, blank)
        dsp.step(frame_matted)
        time1 = time.time() - time1
        tq.set_postfix({"fps": "{:.2f}".format(1.0/time1)})
        tq.update()
    tq.close()

# mask out


def process_no_pre(filename, video_format="MOV"):
    cam = Camera(
        "/home/kie/personal_data/{}.{}".format(filename, video_format))
    dsp = Displayer("output/"+filename+"_no_pre", cam.width, cam.height,
                    cam.fps, show_info=(not args.hide_fps), msg="Filters output only")
    depth_dsp = Displayer("output/"+filename+"_depth", cam.width, cam.height,
                          cam.fps, show_info=(not args.hide_fps), msg="depth map")
    bgr = cv2.imread("/home/kie/personal_data/{}_bgr.png".format(filename))

    if (cam.vertical):
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    v2 = BGv2(args)
    v2.set_bg(bgr)
    mi = MidasSingle(args)
    blank = np.zeros((cam.height, cam.width, 3), dtype='uint8')
    blank.fill(255)
    buf = np.zeros((cam.height, cam.width, 3), dtype='uint8')
    tq = tqdm(total=cam.frame_count)

    dilate_kernel = np.ones((5, 5), dtype='uint8')
    while True:
        buf.fill(0)
        has_next, frame = cam.read()
        if has_next is False:
            break

        time1 = time.time()
        frame_depth = mi.single_frame(frame)
        np.copyto(buf[:, :, 2], frame_depth.cpu().numpy())
        frame_depth_end = filter_depth(frame_depth, 80)
        frame_depth_end = cv2.dilate(
            frame_depth_end, dilate_kernel, iterations=5)
        frame_matted = v2.single_frame(frame)

        frame_matted = apply_mask(frame_matted, frame_depth_end, blank)
        dsp.step(frame_matted)
        depth_dsp.step(buf)
        time1 = time.time() - time1
        tq.set_postfix({"fps": "{:.2f}".format(1.0/time1)})
        tq.update()
    tq.close()

# mask out part of the alpha where depth > threshold


def process_pha(filename, video_format="MOV"):
    cam = Camera(
        "/eva_data/kie/personal_data/{}.{}".format(filename, video_format))
    dsp = Displayer("output/"+filename+"_pha", cam.width, cam.height,
                    cam.fps, show_info=(not args.hide_fps), msg="Filters all")
    bgr = cv2.imread("/eva_data/kie/personal_data/{}_bgr.png".format(filename))

    if (cam.vertical):
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    v2 = BGv2(args)
    v2.set_bg(bgr)
    mi = MidasSingle(args)
    blank = np.zeros((cam.height, cam.width, 3), dtype='uint8')
    blank.fill(255)
    tq = tqdm(total=cam.frame_count)
    dilate_kernel = np.ones((5, 5), dtype='uint8')

    while True:
        has_next, frame = cam.read()
        if has_next is False:
            break

        time1 = time.time()
        frame_depth = mi.single_frame(frame)
        composite, alpha = v2.single_frame(frame)
        frame_mask_pha = filter_depth(frame_depth, 80)
        frame_mask_pha = cv2.dilate(
            frame_mask_pha, dilate_kernel, iterations=5)

        frame_matted = np.zeros(frame.shape, dtype=np.uint8)
        alpha = np.squeeze(alpha.cpu().numpy() * frame_mask_pha)
        # for i in range(3):
        #     frame_matted[:, :, i] = frame[:, :, i] * \
        #         alpha * frame_mask_pha
        frame = np.moveaxis(frame, 2, 0)
        white = 255 * np.ones_like(frame, dtype=np.uint8)
        frame_matted = alpha * frame + \
            (1-alpha) * white
        # frame_matted[:, :, 1] = 255*(1-alpha*frame_mask_pha) * \
        #     (np.ones(frame.shape[:2], dtype=np.uint8))
        output = np.moveaxis(frame_matted, 0, -1).astype(np.uint8)
        dsp.step(output)
        time1 = time.time() - time1
        tq.set_postfix({"fps": "{:.2f}".format(1.0/time1)})
        tq.update()
    tq.close()


def merge(filename, video_format):
    orig = Camera(
        "/home/kie/personal_data/{}.{}".format(filename, video_format))
    all = Camera(
        "/home/kie/research/output/"
        "{}_full.{}".format(filename, "avi"))
    no_pre = Camera(
        "/home/kie/research/output/"
        "{}_no_pre.{}".format(filename, "avi"))
    depth = Camera(
        "/home/kie/research/output/"
        "{}_depth.{}".format(filename, "avi"))

    frames = [None, None, None, None]
    if orig.vertical:
        width = int(orig.width/orig.height*1080)
        output = Displayer("{}_final".format(filename), width*4,
                           1080, orig.fps, show_info=False)
        buf = np.zeros((1080, width*4, 3), dtype='uint8')
    else:
        width = int(1920/2)
        height = int(1080/2)
        output = Displayer("{}_final".format(filename), 1920,
                           1080, orig.fps, show_info=False)
        buf = np.zeros((1080, 1920, 3), dtype='uint8')
    while True:
        s1, frames[0] = orig.read()
        s2, frames[1] = all.read()
        s3, frames[2] = no_pre.read()
        s4, frames[3] = depth.read()
        if (s1 and s2 and s3 and s4) == False:
            break
        if orig.vertical:

            for i in range(4):
                frames[i] = cv2.resize(
                    frames[i], (width, 1080))
                buf[:, width*i:width*(i+1):] = frames[i]

        else:
            for i in range(4):
                frames[i] = cv2.resize(
                    frames[i], (width, height))
            for j in range(2):
                for i in range(2):
                    buf[height*j:height*(j+1), width *
                        i:width*(i+1):] = frames[j*2+i]

        output.step(buf)


if __name__ == "__main__":

    file_h = open(args.file_list, 'r')
    filenames = file_h.readlines()

    for i, ff in enumerate(filenames):
        name, extension = ff.strip("\n").split(' ')

        print("processing", i)
        process_pha(name)
        # process_no_pre(name)
        # merge(name, extension)
        print("finish")

    file_h.close()
