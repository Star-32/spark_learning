# -*- coding: utf-8 -*-
# @shiweitong 2024/4/12

import pandas as pd

from config import X as X1
from another_config import X as X2
from spark_learning import timeit
from spark_learning.utils.models import load_model
from train_new_model import transform_input_raw, multi_model


def gen_submit(modelpath, test_x, submit_file="submit.xlsx"):
    model = load_model(modelpath)

    df = transform_input_raw(pd.read_csv(test_x), "cat_encoder.dill")

    with timeit("predict on test"):
        pred = model.predict(df[X1],df[X1])

    pred_df = pd.DataFrame({"edition_id": df["edition_id"], "pheat": pred})

    with timeit(f"Submit result has been saved to {submit_file}"):
        pred_df.groupby("edition_id").max().to_excel(submit_file)


if __name__ == '__main__':
    from spark_learning import config_logging

    config_logging()

    gen_submit("lgb.dill", "../../data/2023_test_X.csv")
