import cv2
import numpy as np
from tqdm import tqdm
from ps_image_to_array_filter import *


def ps_image_to_array_filter_test(df):
    INPUT_SIZE = 299
    x = np.zeros((len(df), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    for i, file in tqdm(enumerate(df['file'])):
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img_segmented = segment_plant(img)
        img_sharpen = sharpen_image(img_segmented)
        img_resize = cv2.resize(img_sharpen, (INPUT_SIZE, INPUT_SIZE))
        x[i] = img_resize
    return x
