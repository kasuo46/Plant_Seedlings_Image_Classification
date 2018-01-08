import datetime as dt
from ps_load_data import *
from sklearn.preprocessing import LabelBinarizer
from ps_image_to_array import *
from ps_image_to_array_filter import *
from ps_image_to_array_filter_test import *
from ps_fine_tune_part1 import *
from ps_fine_tune_part2 import *
from keras.models import load_model

start = dt.datetime.now()

seed = 1987
sample_per_category = 20
categories_list = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
                   'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
                   'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
num_categories = len(categories_list)
train_dir = 'train/'
test_dir = 'test/'

train_figs_df, train_sample_figs_df, test_figs_df = ps_load_data(train_dir, test_dir, sample_per_category, categories_list)

# print(test_figs_df.head())
# x_train, x_val, y_train, y_val = ps_image_to_array(train_figs_df)
# x_train, x_val, y_train, y_val = ps_image_to_array_filter(train_figs_df)
x_test = ps_image_to_array_filter_test(test_figs_df)

# y_train = LabelBinarizer().fit_transform(y_train)
# y_val = LabelBinarizer().fit_transform(y_val)

# train_score, val_score = ps_fc_fit(bf_train, y_train, bf_val, y_val)
# train_score_ft, val_score_ft = ps_fine_tune_part2(x_train, y_train, x_val, y_val)

model = load_model('data/top_model_part2.h5')
# train_preds = model.predict(x_train/255.0, verbose=1)
# val_preds = model.predict(x_val/255.0, verbose=1)
# train_evaluate = model.evaluate(x_train/255.0, y_train)
# val_evaluate = model.evaluate(x_val/255.0, y_val)
# np.save('data/train_evaluate.npy', train_evaluate)
# np.save('data/val_evaluate.npy', val_evaluate)
# np.save('data/train_preds.npy', train_preds)
# np.save('data/val_preds.npy', val_preds)
# np.save('data/y_train.npy', y_train)
# np.save('data/y_val.npy', y_val)

test_preds = model.predict(x_test/255.0, verbose=1)

test_figs_df['category_id'] = np.argmax(test_preds, axis=1)
test_figs_df['species'] = [categories_list[c] for c in test_figs_df['category_id'].values]
test_figs_df['file'] = test_figs_df['file'].apply(lambda x: x[5:])

test_figs_df[['file', 'species']].to_csv('submission/submission_20180101.csv', index=False)

submit = pd.read_csv('submission/submission_20180101.csv')
print(submit.head())

end = dt.datetime.now()
print('Total time {} s.'.format((end - start).seconds))
