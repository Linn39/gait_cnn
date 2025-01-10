#### gait classification from raw IMU data ####

import os
import json
import yaml
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# import wandb
import plotly.express as px
import plotly.graph_objects as go

from data.DataLoader import DataLoader
from models.train_model import train_model
from models.predict_model import test_model
from visualization.visualize import plot_cv_indices

config = {}

# data
config["dataset"] = "charite"  # "charite" or "duo_gait"
config["sensors"] = ["LF", "RF"]  # ["LF", "RF", "SA"]
config["smoothing"] = "butterworth"  # smooth data using a Butterworth filter
config["segmentation_sensor"] = (
    "RF"  # sensor used for stride segmentation using initial contact  # this can be overwriten by the paretic side in the stroke dataset
)
config["stride_segmentation"] = True  # segment data by strides
config["window_size"] = 1  # window size in strides
config["pad_windows"] = (
    False  # pad the windows to the maximum length. If false: re-sample the windows to the same length
)

# tasks & process
config["overwrite_data"] = True  # overwrite loaded raw data
config["perform_classification"] = (
    True  # False for data visualization (i.e., only plot the raw data from test and train-val splits)
)
config["plot_best_hyperparameters"] = True

# model (note that the hyperparameter search grid may overwrite these values)
config["n_filters"] = 32
config["n_layers"] = 1
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
        "imu0003",
        "imu0006",
        "imu0007",
        "imu0008",
        "imu0009",
        "imu0011",
        "imu0012",
        "imu0013",
    ]

    config["runs"] = ["visit1", "visit2"]
    config["classification_target"] = "visit2"  # one of the runs

    with open("path.json", "r") as f:
        paths = json.load(f)
    original_data_path = paths["data_charite_original"]
    data_base_path = paths["data_charite"]

    # load paretic side from the .csv file
    all_paretic_side = pd.read_csv(
        os.path.join(original_data_path, "raw", "paretic_side.csv")
    )

elif config["dataset"] == "duo_gait":
    sub_list = [
        "sub_01",
        # "sub_02",
        # "sub_03",
        # "sub_05",
        # "sub_06",
        # "sub_07",
        # "sub_08",
        # "sub_09",
        # "sub_10",
        # "sub_11",
        # "sub_12",
        # "sub_13",
        # "sub_14",
        # "sub_15",
        # "sub_17",
        # "sub_18",
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

    # hyperparameter search grid
    grid = {
        # "batch_size": [16, 32],
        # "num_layers": [1, 2],
        # "epochs": [10, 25],
        "num_layers": [1]
    }
    # save the grid to experiment folder
    with open(
        os.path.join(
            data_base_path,
            "results",
            config["experiment_name"],
            "hyperparameter_grid.yaml",
        ),
        "w",
    ) as file:
        yaml.dump(grid, file)

    for sub in sub_list:
        print("\n ====================================")
        print(f"Processing subject {sub}...")

        # set the segmentation sensor and paretic side for the stroke dataset
        both_feet_available = np.logical_and(
            "LF" in config["sensors"], "RF" in config["sensors"]
        )
        if config["dataset"] == "charite":
            if (
                all_paretic_side.loc[
                    all_paretic_side["sub"] == sub, "paretic_side"
                ].values[0]
                == "Right"
            ):
                config["paretic_side"] = "Right"

                if both_feet_available:
                    # overwrite only if both LF and RF sensors are available
                    config["segmentation_sensor"] = "RF"
            else:
                if (
                    all_paretic_side.loc[
                        all_paretic_side["sub"] == sub, "paretic_side"
                    ].values[0]
                    == "Left"
                ):
                    config["paretic_side"] = "Left"
                else:
                    config["paretic_side"] = "None"

                # use the left foot sensor as default
                if both_feet_available:
                    # overwrite only if both LF and RF sensors are available
                    config["segmentation_sensor"] = "LF"

        # create folder to save the results for this sub
        sub_results_dir = os.path.join(results_dir, sub)
        os.makedirs(sub_results_dir, exist_ok=True)

        # save the subject specific config as json
        with open(os.path.join(results_dir, sub, "config_sub.json"), "w") as f:
            json.dump(config, f, indent=4)

        # get data from one subject
        sub_data = all_data.loc[all_data["sub"] == sub].reset_index(drop=True)

        # split data to get the test set
        test_ratio = 0.15  # ratio of test data
        skf_test = StratifiedKFold(
            int(1 / test_ratio),
            shuffle=False,
        )  # data split to get the test set
        splits = skf_test.split(sub_data, sub_data[config["classification_target"]])

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

        train_model(
            config=config,
            train_val_data=train_val_data,
            # save_path=results_dir,
            original_data_path=original_data_path,
            data_base_path=data_base_path,
            data_exploration_dir=data_exploration_dir,
            sub=sub,
        )

        # test the model on the test set
        test_model(dataset=config["dataset"], exp_name=experiment_name, sub=sub)

if config["plot_best_hyperparameters"]:
    # plot the best hyperparameters from all best models
    # experiment_name = "2024-03-06_11-28-37"  # select a previous experiment
    best_hyperparameters = []  # collect the best hyperparameters from all best modell
    for sub in sub_list:
        # Load the hyperparameters from the .yaml files
        hyperparameter_dir = os.path.join(
            data_base_path, "results", experiment_name, sub
        )
        with open(os.path.join(hyperparameter_dir, "best_params.yaml"), "r") as stream:
            try:
                dict = yaml.safe_load(stream)
                dict["sub"] = sub
                best_hyperparameters.append(dict)
            except yaml.YAMLError as exc:
                print(exc)

    # Convert the list of dictionaries to a DataFrame
    hyper_param_df = pd.DataFrame(best_hyperparameters)

    # Convert 'sub' column to numerical values
    le = LabelEncoder()
    hyper_param_df["sub_encoded"] = le.fit_transform(hyper_param_df["sub"])

    # Create the plot
    fig = px.parallel_coordinates(hyper_param_df, color="sub_encoded")

    # Update the layout
    fig.update_layout(
        autosize=False,
        width=700,
        height=400,
        title=f"Best hyperparameters for all subjects\n {experiment_name}",
        margin=go.layout.Margin(t=100),  # Increase top margin
    )

    fig.show()
    fig.write_image(
        os.path.join(
            data_base_path, "results", experiment_name, "all_best_hyperparameters.pdf"
        )
    )
