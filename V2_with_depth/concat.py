# This is the custom code of V2 with depth info


import cv2
import numpy as np
import os
import glob
import torch
from MiDaS import utils
import PIL

from torchvision.transforms import Compose
from MiDaS.midas.midas_net import MidasNet
from MiDaS.midas.midas_net_custom import MidasNet_small
from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

model_path = "/kaggle/input/weights-for-v2-and-intel/intel-model-f6b98070.pt"
test_data_path = "/kaggle/input/matting-test-video/personal data/"

video_list_h = []
video_list_v = ["close"]

threshold = 100
device = None

def concat(video_path, bgr_path, name="output", vertical: bool = False):

    model, transform = initialize(model_path)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    bgr_cap = cv2.imread(bgr_path)
    
    # Check if camera opened successfully
    if (not cap.isOpened()):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("video shape", frame_width, frame_height)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(
        *'MPEG'), int(cap.get(cv2.CAP_PROP_FPS)), (frame_width, frame_height))
    buf = np.zeros((1920,1080, 3), dtype='uint8')

    while(True):
        ret1, frame1 = cap.read()

        if ret1 == False:
            break
        else:
            if (vertical):
                pass
                # frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
                # print("vertical")
            # print(frame.shape)

        # Processing starts
        res1 = computeDepth(model, transform, frame1)
        print(frame1.shape)
        with np.nditer(res1, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = 1 if x > threshold else 0

        for i in range(3):
            buf[:,:,i] = (res1 * frame1[:,:,i]) + bgr_cap - (res1 * bgr_cap[:,:,i])
        out.write(buf)
        # cv2.imshow('frame',buf)
        # cv2.waitKey(100)

    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    # cv2.destroyAllWindows()


def initialize(model_path, optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        model_path (str): path to saved model
    """
    print("initialize")

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    model = MidasNet(model_path, non_negative=True)
    net_w, net_h = 384, 384

    transform = Compose(
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

    model.eval()

    if optimize == True:
        rand_example = torch.rand(1, 3, net_h, net_w)
        model(rand_example)
        traced_script_module = torch.jit.trace(model, rand_example)
        model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    return model, transform


def write_depth(depth, bits=1):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if bits == 1:
        return out.astype("uint8")
    elif bits == 2:
        return out.astype("uint16")

    raise RuntimeError("Invalid bits")


def computeDepth(model, transform, img, optimize: bool = True):
    global device
    print("start processing")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]
    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
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
    
    output = write_depth(prediction, bits=1)

    print("finished")
    return output


def side_by_side(images: list):

    imgs = [PIL.Image.open(i) for i in images]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))

    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save('Trifecta.jpg')


if (__name__ == "__main__"):
    for i in video_list_h:
        video_path = test_data_path + i + ".MOV"
        bgr_path = test_data_path + i + "_bgr.png"
        concat(video_path, bgr_path, i, False)
    for i in video_list_v:
        video_path = test_data_path + i + ".MOV"
        bgr_path = test_data_path + i + "_bgr.png"
        concat(video_path, bgr_path, i, True)
