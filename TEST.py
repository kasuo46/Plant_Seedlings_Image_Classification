# from sklearn.preprocessing import LabelBinarizer
import numpy as np

# a = np.array([1, 4, 5, 3, 7, 6, 6, 0, 2, 2, 3, 7])
# # print(a.shape)
# #
# # print(LabelBinarizer().fit_transform(a).shape)
#
# # print(np.random.choice(np.arange(0, 100), 10, replace=False))
#
# b = np.array([1, 4, 4, 3, 7, 6, 6, 0, 2, 2, 3, 7])
#
# print(np.sum(a != b))

# train_preds = np.load('data/train_preds.npy')
# val_preds = np.load('data/val_preds.npy')
# y_train = np.load('data/y_train.npy')
# y_val = np.load('data/y_val.npy')

# print('stop')
# train_evaluate = np.load('data/train_evaluate.npy')
# val_evaluate = np.load('data/val_evaluate.npy')
# print(train_evaluate, val_evaluate)

# val_preds = np.load('data/val_preds.npy')
# print(val_preds.shape)
# print(val_preds[0:5])
# val_labels = np.argmax(val_preds, axis=1)
# print(val_labels.shape)
# print(val_labels)

print(np.load('data/train_evaluate.npy'))
print(np.load('data/val_evaluate.npy'))
