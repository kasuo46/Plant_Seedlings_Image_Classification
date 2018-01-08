import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def ps_image_to_array_filter(df):
    INPUT_SIZE = 299
    x = np.zeros((len(df), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    y = df['category_id'].values.reshape((len(df),))
    for i, file in tqdm(enumerate(df['file'])):
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img_segmented = segment_plant(img)
        img_sharpen = sharpen_image(img_segmented)
        img_resize = cv2.resize(img_sharpen, (INPUT_SIZE, INPUT_SIZE))
        x[i] = img_resize
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # np.save('data/x_train.npy', x_train/255.)
    # np.save('data/x_val.npy', x_val/255.)
    # np.save('data/y_train.npy', y_train)
    # np.save('data/y_val.npy', y_val)
    return x_train, x_val, y_train, y_val


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask=mask)
    return output


def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
