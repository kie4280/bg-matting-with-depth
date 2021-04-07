import os
import glob
import cv2
from cv2 import data
import requests
import asyncio
import functools
from bing_image_downloader import downloader
import random
import numpy as np
import pathlib



search_terms = ["house", "airport", "cat", "school building"]


class RandomData:
    def __init__(self, search_terms, category_size=100) -> None:
        self.category_size = category_size
        self.loop = asyncio.get_event_loop()
        self.search_terms = search_terms

    def generate_data(self, dataset_size=100):
        old_files = glob.glob("dataset/output/*")
        for f in old_files:
            os.remove(f)
        tasks = []
        for i in self.search_terms:
            task = self.loop.create_task(self._getbackground(i))
            tasks.append(task)
        fu = asyncio.wait(tasks)
        self.loop.run_until_complete(fu)
        # self._random_place(dataset_size)

    async def _getbackground(self, search_term):
        # creating object
        await self.loop.run_in_executor(None, functools.partial(
            downloader.download,
            search_term, limit=self.category_size,  output_dir='dataset',
            adult_filter_off=True, force_replace=False, timeout=60, show_progress=False))

    def _random_place(self, dataset_size):

        foregrounds = glob.glob("dataset/foreground/*.png")
        backgrounds = []

        for i in self.search_terms:
            f = glob.glob("dataset/{}/**".format(i))
            backgrounds.extend(f)

        for i in range(dataset_size):
            which_back = random.randrange(0, len(backgrounds))

            which_fore = random.randrange(0, len(foregrounds))
            bg = cv2.imread("{}".format(
                backgrounds[which_back]), cv2.IMREAD_UNCHANGED)
            fg = cv2.imread("{}".format(
                foregrounds[which_fore]), cv2.IMREAD_UNCHANGED)
            bg_shape = np.asarray(np.shape(bg)[0:2]) * 0.8
            fg_shape = np.asarray(np.shape(fg)[0:2])
            scale_ratio = np.max(fg_shape / bg_shape)

            if scale_ratio > 1.0:
                fg = cv2.resize(fg, dsize=(
                    fg_shape[1], fg_shape[0]), interpolation=cv2.INTER_AREA)


class AlphaExtractor:
    def __init__(self, input_dir, output_dir: str = ".") -> None:
        self.input_dir = pathlib.Path(input_dir)
        self.output_dir = pathlib.Path(output_dir)

    def extract(self):
        
        for i in self.input_dir.iterdir():
            img = cv2.imread(str(i), cv2.IMREAD_UNCHANGED)
            buf = np.ndarray(np.shape(img)[0:2], dtype=np.uint8)
            buf = img[:,:,3]
            cv2.imwrite(str(self.output_dir.joinpath(i.name)), buf)
        pass


class VideoSlicer:
    def __init__(self, input_video:str, output_dir:str = "./output") -> None:
        self.input_video = pathlib.Path(input_video)
        self.output_dir = pathlib.Path(output_dir)
    def start(self):
        cap = cv2.VideoCapture(str(self.input_video))
        frame_index:int = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(str(self.output_dir.joinpath(self.input_video.stem)) + "-" + str(frame_index) + ".png", frame)
            print("Processed frame {}".format(frame_index))
            frame_index += 1

        

if __name__ == "__main__":
    # rd = AlphaExtractor("dataset/foreground", "dataset/alpha")
    # rd.extract()
    vs = VideoSlicer("data/custom_videos/lib_3.mp4", output_dir="data/custom")
    vs.start()
