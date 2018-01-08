import os
import pandas as pd


def ps_load_data(train_dir, test_dir, sample_per_category, categories_list):
    test_figs_list = os.listdir(test_dir)
    test_figs_df = pd.DataFrame(test_figs_list, columns=['file'])
    test_figs_df['file'] = test_dir + test_figs_df['file']

    train_figs_list = []
    for category_id, category in enumerate(categories_list):
        for file in os.listdir(train_dir + category + '/'):
            train_figs_list.append([train_dir + '{}/{}'.format(category, file),
                                    category_id, category])
    train_figs_df = pd.DataFrame(train_figs_list, columns=['file', 'category_id', 'category'])

    train_sample_figs_list = []
    for category_id, category in enumerate(categories_list):
        for file in os.listdir(train_dir + category + '/')[:sample_per_category]:
            train_sample_figs_list.append([train_dir + '{}/{}'.format(category, file),
                                           category_id, category])
    train_sample_figs_df = pd.DataFrame(train_sample_figs_list,
                                        columns=['file', 'category_id', 'category'])

    return train_figs_df, train_sample_figs_df, test_figs_df
