import os

import cv2
from natsort import natsorted


def read_video_with_image_pairs(path, size=None):
    ir_path = os.path.join(path, "ir")
    vi_path = os.path.join(path, "vi")
    ir_list_dir = os.listdir(ir_path)
    vi_list_dir = os.listdir(vi_path)
    img_pairs = natsorted(list(set(ir_list_dir).intersection(set(vi_list_dir))))
    for img_pair in img_pairs:
        ir_image = cv2.imread(os.path.join(ir_path, img_pair))
        vi_image = cv2.imread(os.path.join(vi_path, img_pair))
        if size is not None:
            ir_image = cv2.resize(ir_image, size)
            vi_image = cv2.resize(vi_image, size)
        yield ir_image, vi_image
