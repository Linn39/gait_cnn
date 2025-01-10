#### gait classification from raw IMU data ####

import os
import json
import pickle
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from data.DataLoader import DataLoader
from features.FeatureBuilder import FeatureBuilder
from models.train_model import (
    get_CNN_model,
    execute_training,
    normalize_3d_data,
)
from models.predict_model import test_model
from visualization.visualize import plot_cv_indices

config = {}

# data
config["dataset"] = "charite"  # "charite" or "duo_gait"
config["sensors"] = ["LF"]
config["smoothing"] = "butterworth"  # smooth data using a Butterworth filter
config[
    "segmentation_sensor"
] = "LF"  # sensor used for stride segmentation using initial contact
config["stride_segmentation"] = True  # segment data by strides
config["window_size"] = 1  # window size in strides

# tasks & process
config["overwrite_data"] = False  # overwrite loaded raw data
config["perform_classification"] = True  # False for data visualization

# model
config["initial_lr"] = 0.02  # initial learning rate
config["batch_size"] = 32  # batch size
config["epochs"] = 25  # number of epochs
config["early_stopping"] = True  # use early stopping
config["reduce_lr"] = True  # reduce learning rate on plateau
config["evaluation_metric"] = "AUC"  # "accuracy"  # metric to use for model evaluation
config["use_wandb"] = False  # use wandb for logging

# dataset specific
if config["dataset"] == "charite":
    sub_list = [
        "imu0001",
        "imu0002",
        # "imu0003",
        # "imu0006",
        # "imu0007",
        # "imu0008",
        # "imu0009",
        # "imu0011",
        # "imu0012",
        # "imu0013",
    ]

    config["runs"] = ["visit1", "visit2"]
    config["classification_target"] = "visit2"  # one of the runs

    with open("path.json", "r") as f:
        paths = json.load(f)
    original_data_path = paths["data_charite_original"]
    data_base_path = paths["data_charite"]

elif config["dataset"] == "duo_gait":
    sub_list = [
        "sub_01",
        "sub_02",
        "sub_03",
        "sub_05",
        "sub_06",
        "sub_07",
        "sub_08",
        "sub_09",
        "sub_10",
        "sub_11",
        "sub_12",
        "sub_13",
        "sub_14",
        "sub_15",
        "sub_17",
        "sub_18",
    ]

    config["runs"] = [
        "OG_st_control",
        "OG_st_fatigue",
    ]
    config["classification_target"] = "OG_st_fatigue"  # one of the runs

    with open("path.json", "r") as f:
        paths = json.load(f)
    original_data_path = paths["data_duo_gait_original"]
    data_base_path = paths["data_duo_gait"]

# folder to save data exploration plots
data_exploration_dir = os.path.join(data_base_path, "data_exploration")
if not os.path.exists(data_exploration_dir):
    os.makedirs(data_exploration_dir)

if config["overwrite_data"] or not os.path.exists(
    os.path.join(data_base_path, "all_raw_data.csv")
):
    # load all data
    data_loader = DataLoader(
        original_data_path=original_data_path,
        data_base_path=data_base_path,
        subs=sub_list,
        config=config,
    )
    all_data = data_loader.get_all_data()

    # save raw data
    all_data.to_csv(os.path.join(data_base_path, "all_raw_data.csv"), index=False)

else:
    # load raw data
    all_data = pd.read_csv(os.path.join(data_base_path, "all_raw_data.csv"))

if config["perform_classification"]:
    # create a folder to save the results
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["experiment_name"] = experiment_name
    results_dir = os.path.join(data_base_path, "results", experiment_name)
    os.makedirs(results_dir)

    # save the config
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

for sub in sub_list:
    print("\n ====================================")
    print(f"Processing subject {sub}...")

    # get data from one subject
    sub_data = all_data.loc[all_data["sub"] == sub].reset_index(drop=True)

    # split data to get the test set
    test_ratio = 0.1  # ratio of test data
    skf_test = StratifiedKFold(
        int(1 / test_ratio),
        shuffle=False,
    )  # data split to get the test set
    splits = skf_test.split(sub_data, sub_data[config["classification_target"]])

    #### debugging ####
    # # plot train-test split
    # fig, ax = plt.subplots()
    # plot_cv_indices(
    #     skf_test,
    #     sub_data,
    #     sub_data[config["classification_target"]],
    #     sub_data["sub"],
    #     ax,
    #     int(1 / test_ratio),
    # )
    # plt.show()
    #### end debugging ####

    # get the test data indices
    train_val_indices, test_indices = list(splits)[
        int(0.5 * (1 / test_ratio))
    ]  # get the middle split as test data
    test_data = sub_data.iloc[test_indices]

    # save the test data
    test_data_dir = os.path.join(data_base_path, "test_data", sub)
    if not os.path.exists(test_data_dir):
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)
    test_data.to_csv(os.path.join(test_data_dir, "test_data.csv"), index=False)

    # use the rest of the data for training and validation
    train_val_data = sub_data.iloc[train_val_indices]
    # save the train_val data
    train_val_data.to_csv(
        os.path.join(test_data_dir, "train_val_data.csv"), index=False
    )

    # split the rest of the data into training and validation sets
    skf_train_val = StratifiedKFold(
        5, shuffle=False
    )  # 5-fold cross-validation for train/validation split

    best_model_n = 0  # the best model number, 0 as default
    best_val_score = 0  # the best validation metric, 0 as default
    for i, (train_index, val_index) in enumerate(
        skf_train_val.split(
            train_val_data, train_val_data[config["classification_target"]]
        )
    ):
        if config["perform_classification"]:
            # create a folder to save the results in that fold
            fold_results_dir = os.path.join(results_dir, sub, f"fold_{i}")
            os.makedirs(fold_results_dir)

        # get raw train and val data (time series before windowing)
        X_y_train_raw = train_val_data.iloc[train_index]
        X_y_val_raw = train_val_data.iloc[val_index]

        # get stride windows for train data
        feature_builder_train = FeatureBuilder(
            original_data_path=original_data_path,
            data_base_path=data_base_path,
            data_df=X_y_train_raw,
            config=config,
            scaler=None,  # create new scaler for training data
        )
        feature_builder_train.normalize_2d_data(plot_title=f"{sub} train data")
        train_scaler = feature_builder_train.get_scaler()
        X_train, y_train = feature_builder_train.make_windows()
        feature_builder_train.plot_all_windows(
            sensor=config["sensors"][0],
            sub=sub,
            save_fig_path=os.path.join(data_exploration_dir, "train_windows"),
        )

        # get stride windows for validation data
        feature_builder_val = FeatureBuilder(
            original_data_path=original_data_path,
            data_base_path=data_base_path,
            data_df=X_y_val_raw,
            config=config,
            scaler=train_scaler,  # use the same scaler as the training data
        )
        feature_builder_val.normalize_2d_data(plot_title=f"{sub} val data")
        X_val, y_val = feature_builder_val.make_windows(
            max_window_size=X_train.shape[1]
        )  # use the same window size as the training data
        feature_builder_val.plot_all_windows(
            sensor=config["sensors"][0],
            sub=sub,
            save_fig_path=os.path.join(data_exploration_dir, "val_windows"),
        )

        #### debugging: plot windows ####
        # plot_windows(
        #     data=[X_val, y_val],
        #     feature=f"AccY_{config['sensors'][0]}",
        # )
        #### end debugging ####

        if config["perform_classification"]:
            # save the scaler
            scaler_path = os.path.join(fold_results_dir, "scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(train_scaler, f)

            # train the model
            history, model = execute_training(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config=config,
                save_path=fold_results_dir,
            )
            # get validation metric from history
            metric = config["evaluation_metric"]
            if metric == "AUC":
                metric = "auc"  # convert to lowercase
            val_metric = history.history["val_" + metric][-1]
            if val_metric > best_val_score:
                best_val_score = val_metric
                best_model_n = i

    if config["perform_classification"]:
        # log the best model number in config
        config["best_model_n_" + sub] = best_model_n
        with open(os.path.join(results_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # test the model on the test set
        test_model(exp_name=experiment_name, sub=sub)
