# -*- coding: utf-8 -*-
# @shiweitong 2024/4/12
import logging
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, cv, Dataset, log_evaluation
from xgboost import XGBRegressor
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from spark_learning.CatEncoder import CatEncoder
from spark_learning.utils.models import save_model
from config import X as X1
from config import y
from another_config import X as X2


# transform categorical data (which are usually objects or strings) into numerical values
def transform_input_raw(dataset: pd.DataFrame, encoder: (str, CatEncoder), with_label=True, update_encoder=False):
    cat_features = [
        'entity_country', 'entity_region', 'genre', 'sub_genre', 'tag_list',
        'developer_id', 'developer', 'publisher_id', 'publisher',
    ]
    if not isinstance(encoder, CatEncoder) and (update_encoder or not os.path.exists(encoder)):
        _encoder = CatEncoder()
        for fea in cat_features:
            dataset[fea] = _encoder.fit_transform(
                fea,
                dataset[fea]
                # dataset[fea].apply(lambda x: (x.split("|")[0] if "|" in x else x) if isinstance(x, str) else x)
            )
        _encoder.save(encoder)
    else:
        encoder = CatEncoder.load(encoder) if isinstance(encoder, str) else encoder
        for fea in cat_features:
            dataset[fea] = encoder.transform(
                fea,
                dataset[fea]
                # dataset[fea].apply(lambda x: (x.split("|")[0] if "|" in x else x) if isinstance(x, str) else x)
            )

    time_features = [
        "date","release_time"
    ]
    for fea in time_features:
        dataset[fea] = (pd.to_datetime(dataset['release_time']).dt.tz_localize(None) - pd.to_datetime(dataset[fea])
                        ).dt.days

    for fea in dataset.columns:
        if dataset[fea].dtype != 'object':
            new_column = dataset.groupby('edition_id')[fea].transform('mean')
            dataset[fea].fillna(new_column)

    dataset = dataset.sort_values(by=['edition_id','date'])
    dataset['mean_wishlist_rank'] = dataset.groupby('edition_id')['wishlist_rank'].transform('mean')
    dataset['change_wishlist_rank'] = dataset.wishlist_rank-dataset.mean_wishlist_rank
    dataset['wishlist_rank_diff'] = dataset.groupby('edition_id')['wishlist_rank'].diff()
    dataset['wishlist_rank_change_rate'] = dataset['wishlist_rank_diff'] / dataset.groupby('edition_id')['wishlist_rank'].shift(1)
    dataset['cat_pcu_mean'] = dataset.groupby('genre')['EA_pcu'].transform('median')
    dataset['cat_pcu_val'] = dataset['EA_pcu'] / dataset['cat_pcu_mean']
    dataset['cat_pcu_val'] = dataset['cat_pcu_val'].fillna(dataset['cat_pcu_val'].median())
    scaler = MinMaxScaler()
    subdf = dataset['cat_pcu_val'].to_frame(name='p')
    dataset['cat_pcu_val'] = scaler.fit_transform(subdf)
    return dataset

class multi_model():
    def __init__(self, model1, model2) -> None:
        self.model1 = model1
        self.model2 = model2
    
    def fit(self, x_train1, y_train1, x_eval1, y_eval1, x_train2, y_train2, x_eval2, y_eval2):
        self.model1.fit(x_train1, y_train1, eval_set=[(x_eval1, y_eval1)],
            eval_metric='rmse', callbacks=[early_stopping(stopping_rounds=50),])
        self.model2.fit(x_train2, y_train2, eval_set=[(x_eval2, y_eval2)])

    def predict(self, x1, x2):
        res1 = self.model1.predict(x1)
        res2 = self.model2.predict(x2)
        res = (res1 * 0.8 + res2 * 0.2) 
        return res

def train_pheat_demo():
    logging.info("loading training data")
    # dataset_1 = pd.read_excel("../../data/40_train.xlsx")
    # dataset_2 = pd.read_excel("../../data/train.xlsx")
    # dataset = pd.concat([dataset_1,dataset_2])
    dataset = pd.read_csv("../../data/2023_partial_train.csv")

    dataset = transform_input_raw(dataset, "cat_encoder.dill", update_encoder=True)
    main_dataset = dataset[~dataset["wishlist_rank"].isna()]
    supplement_dataset = dataset[dataset["wishlist_rank"].isna()]
    main_games = main_dataset["edition_id"].unique()
    supplement_games = supplement_dataset["edition_id"].unique()

    dataset['main_supplement'] = np.where(~(dataset['wishlist_rank'].isna()), 'main', 'supplement')
    dataset['combined_category'] = dataset['main_supplement'] + '_' + dataset['genre'].astype(str)
    all_games = np.concatenate([main_games, supplement_games])
    # combined_labels = dataset.set_index('edition_id').loc[all_games]['combined_category'].values
    logging.info(f"main games: {len(main_games)}; supplement games: {len(supplement_games)}")
    train_ds, eval_ds = train_test_split(
        dataset,
        stratify=dataset['combined_category'],
        test_size=0.1,
        random_state=42,
    )
    # logging.info(f"games for train: {len(games_train)}, games for test: {len(games_test)}")

    # train_ds = dataset[dataset["edition_id"].isin(games_train)]
    # eval_ds = dataset[dataset["edition_id"].isin(games_test)]

    train_ds2 = train_ds[train_ds['EA_pcu']<=4]
    train_ds1 = train_ds[~(train_ds['EA_pcu']<=4)]
    eval_ds2 = eval_ds[eval_ds['EA_pcu']<=4]
    eval_ds1 = eval_ds[~(eval_ds['EA_pcu']<=4)]

    logging.info(f"Train: {len(train_ds)}, Eval {len(eval_ds)}")

    x_train1_all, x_train2_all, y_train = train_ds[X1], train_ds[X2], train_ds[y]
    x_eval1_all, x_eval2_all, y_eval = eval_ds[X1], eval_ds[X2], eval_ds[y]
    x_train1, y_train1 = train_ds[X1], train_ds[y]
    x_eval1, y_eval1 = eval_ds[X1], eval_ds[y]
    x_train2, y_train2 = train_ds2[X1], train_ds2[y]
    x_eval2, y_eval2 = eval_ds[X1], eval_ds[y]


    model1 = LGBMRegressor(
        num_leaves=100,
        max_depth=8,
        min_child_samples=5000,
        objective='regression',
        learning_rate=0.01,
        n_estimators=500,
        verbosity=2,
        subsample = 0.9, 
        random_state=42,
    )
    
    model2 = XGBRegressor(
        objective = 'reg:squarederror',
        eval_metric = 'rmse',
        eta = 0.02, # Learning rate
        n_estimators = 100,
        min_child_weight = 1000,  # Increase to avoid overfitting
        max_depth = 7,  # Limit the depth of the tree
        subsample = 1.0, 
        colsample_bytree = 1.0, 
        seed = 42
    )

    bigmodel = multi_model(model1,model2)
    bigmodel.fit(x_train1,y_train1,x_eval1,y_eval1,x_train2,y_train2,x_eval2,y_eval2)
    
    train_pred = bigmodel.predict(x_train1_all, x_train1_all)
    eval_pred = bigmodel.predict(x_eval1_all, x_eval1_all)

    logging.info({
        "ndcg_score_train": ndcg_score(np.expand_dims(y_train, 0), np.expand_dims(train_pred, 0)),
        "ndcg_score_eval": ndcg_score(np.expand_dims(y_eval, 0), np.expand_dims(eval_pred, 0)),
    })
    
    save_model(bigmodel, "lgb.dill")


if __name__ == '__main__':
    from spark_learning.utils import config_logging

    config_logging()

    train_pheat_demo()
