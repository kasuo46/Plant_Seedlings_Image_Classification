from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def ps_image_to_array(df):
    INPUT_SIZE = 299
    x = np.zeros((len(df), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
    y = df['category_id'].values.reshape((len(df),))
    for i, file in tqdm(enumerate(df['file'])):
        img = load_img(file, target_size=(INPUT_SIZE, INPUT_SIZE))
        img = img_to_array(img)
        x[i] = img
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    # np.save('data/x_train.npy', x_train/255.)
    # np.save('data/x_val.npy', x_val/255.)
    # np.save('data/y_train.npy', y_train)
    # np.save('data/y_val.npy', y_val)
    return x_train, x_val, y_train, y_val
