import os
import glob
import cv2
from cv2 import data
import requests
import asyncio
import functools
from bing_image_downloader import downloader
import random


search_terms = ["house", "airport", "cat", "school building"]


class RandomData:
    def __init__(self, search_terms, category_size=100) -> None:
        self.category_size = category_size
        self.loop = asyncio.get_event_loop()
        self.search_terms = search_terms
    def generate_data(self, dataset_size = 100):
        old_files = glob.glob("dataset/output/*")
        for f in old_files:
            os.remove(f)
        tasks = []
        for i in self.search_terms:
            task = self.loop.create_task(self._getbackground(i))
            tasks.append(task)
        fu = asyncio.wait(tasks)
        self.loop.run_until_complete(fu)
        self._random_place(dataset_size)

    async def _getbackground(self, search_term):
        # creating object
        await self.loop.run_in_executor(None, functools.partial(
            downloader.download,
            search_term, limit=self.category_size,  output_dir='dataset',
            adult_filter_off=True, force_replace=False, timeout=60))

    def _random_place(self, dataset_size):
        foregrounds = []
        fs = glob.glob("dataset/foreground/*.png")
        for i in fs:
            foregrounds.append(i)
        for i in range(dataset_size):
            which_cat = random.randrange(0, len(self.search_terms))
            in_cat = random.randint(1, self.category_size)
            which_fore = random.randrange(0, len(foregrounds))
            bg = cv2.imread("dataset/{}/Image_{}.jpg".format(which_cat, in_cat))
            fg = cv2.imread("dataset/foreground/{}".format(foregrounds[which_fore]))
        

class VideoSlicer:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    rd = RandomData(search_terms=search_terms, dataset_size=10)
    rd.generate_data()
