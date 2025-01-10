import os
import json
import glob
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# add the src directory to the path
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from features.FeatureBuilder import FeatureBuilder
from models.train_model import normalize_3d_data


def test_model(dataset: str, exp_name: str, sub: str):
    """
    Predicts the test data using the best model from the train-val folds.

    Parameters
    ----------
    exp_name: the experiment directory
    sub: the subject to predict

    Returns: None
    -------

    """
    # load test data
    with open("path.json", "r") as f:
        paths = json.load(f)
    original_data_path = paths[f"data_{dataset}_original"]
    data_base_path = paths[f"data_{dataset}"]
    data_exploration_dir = os.path.join(data_base_path, "data_exploration")

    # load the subject-specific config with segmentation sensor and paretic side
    config= json.load(
        open(os.path.join(data_base_path, "results", exp_name, sub, "config_sub.json"), "r")
    )

    best_model_dir = os.path.join(data_base_path, "results", exp_name, sub)
    model_path = os.path.join(best_model_dir, "best_model.h5")
    model = tf.keras.models.load_model(model_path)

    # load the test data
    test_data_dir = os.path.join(data_base_path, "test_data", sub)
    test_data = pd.read_csv(os.path.join(test_data_dir, "test_data.csv"))

    # #### load all scalers from the training data and aggregate them
    # scaler_paths = glob.glob(
    #     os.path.join(data_base_path, "results", exp_name, sub, "scaler_fold_*.pkl")
    # )
    # scales = []
    # means = []
    # for scaler_path in scaler_paths:
    #     with open(scaler_path, "rb") as f:
    #         scaler = pickle.load(f)
    #         scales.append(scaler.scale_)
    #         means.append(scaler.mean_)

    # # Calculate the average scale and mean
    # avg_scale = np.mean(scales, axis=0)
    # avg_mean = np.mean(means, axis=0)

    # # Create a new scaler with the average scale and mean
    # avg_scaler = StandardScaler()
    # avg_scaler.scale_ = avg_scale
    # avg_scaler.mean_ = avg_mean
    # avg_scaler.n_features_in_ = len(avg_scale)  # needed for innvestigate

    # # save the scaler
    # avg_scaler_path = os.path.join(best_model_dir, "avg_scaler.pkl")
    # with open(avg_scaler_path, "wb") as f:
    #     pickle.dump(avg_scaler, f)

    # load the train_val scaler used to train the final best model
    train_val_scaler_path = os.path.join(best_model_dir, "train_val_scaler.pkl")
    with open(train_val_scaler_path, "rb") as f:
        train_val_scaler_dict = pickle.load(f)

    # cut the test data into windows
    feature_builder = FeatureBuilder(
        original_data_path=original_data_path,
        data_base_path=data_base_path,
        data_df=test_data,
        config=config,
        scaler_dict=train_val_scaler_dict,  # use the average scaler from all folds
    )

    feature_builder.normalize_2d_data(plot_title=f"{sub} test data")
    max_window_size = model.input_shape[
        1
    ]  # get the window size from the input shape of the model
    test_X, test_y, _ = feature_builder.make_windows(max_window_size=max_window_size)
    feature_builder.plot_all_windows(
        sensors=config["sensors"],
        sub=sub,
        save_fig_path=os.path.join(data_exploration_dir, "test_windows"),
    )

    # predict the test data
    y_pred = model.predict(test_X)

    # save the predictions
    predictions_dir = os.path.join(best_model_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    predictions_path = os.path.join(predictions_dir, "predictions.csv")
    pd.DataFrame(y_pred).to_csv(predictions_path, index=False)

    # evaluate the predictions using metrics from config
    y_pred = tf.argmax(y_pred, axis=1)

    if len(test_y.shape) == 1:
        test_y = test_y[:, None]
    elif len(test_y.shape) == 2:  # if two dimensions (i.e., one-hot coded label)
        # undo the one-hot encoding
        test_y = np.argmax(test_y, axis=1)
    confusion_mat = confusion_matrix(y_true=test_y, y_pred=y_pred)
    print("\n test confusion matrix:\n", confusion_mat)

    evaluation_results = {}

    if config["evaluation_metric"] == "accuracy":
        accuracy = tf.keras.metrics.Accuracy()
        accuracy.update_state(y_true=test_y, y_pred=y_pred)
        evaluation_results["accuracy"] = accuracy.result().numpy().round(4)
        print(
            f"Test accuracy on model from fold {config[f'best_model_n_{sub}']}:"
            + f" {evaluation_results['accuracy']}"
        )

        f1 = f1_score(test_y, y_pred, average="weighted")
        evaluation_results["f1_score"] = f1.round(4)
        print(
            f"Test F1 score on model from fold {config[f'best_model_n_{sub}']}:"
            + f" {evaluation_results['f1_score']}"
        )
    elif config["evaluation_metric"] == "AUC":
        auc = tf.keras.metrics.AUC()
        auc.update_state(y_true=test_y, y_pred=y_pred)
        evaluation_results["auc"] = auc.result().numpy().round(4)
        print(f"Test AUC on best model:" + f" {evaluation_results['auc']}")
    else:
        raise ValueError(
            f"evaluation metric {config['evaluation_metric']} not supported"
        )

    # save the evalution results
    evaluation_results_df = pd.DataFrame.from_dict(
        evaluation_results, orient="index"
    ).transpose()

    # add sub info
    evaluation_results_df["sub"] = sub
    # add sensors used for classification
    if len(config["sensors"]) == 1:
        sensors_str = config["sensors"][0]
    else:
        sensors_str = "_".join(config["sensors"])
    evaluation_results_df["sensors"] = sensors_str

    # create a csv file to write the results to
    predictions_path = os.path.join(
        data_base_path, "results", exp_name, "test_results.csv"
    )
    if not os.path.exists(predictions_path):
        # create new file with column names
        evaluation_results_df.to_csv(predictions_path, index=False, header=True)
    else:
        # append to existing file
        evaluation_results_df.to_csv(
            predictions_path, index=False, mode="a", header=False
        )


def get_fold_val_data(fold_n: int, sub: str, config: dict, data_base_path: str):
    """get data from the nth fold in the train-validation data

    Args:
        exp_name (str): name of the experiment
        fold_n (int): nth fold
        sub (str): subject to get the data from
        config (dict): config dictionary
        data_base_path (str): path to the data directory

    Returns:
        DataFrame: val data in nth fold
    """
    # load train validation data for the subject
    train_val_data = pd.read_csv(
        os.path.join(data_base_path, "test_data", sub, "train_val_data.csv")
    )
    # split data to get the validation set
    skf_test = StratifiedKFold(5, shuffle=False)  # 5-fold cross-validation
    splits = skf_test.split(
        train_val_data, train_val_data[config["classification_target"]]
    )

    # get the val data indices from the nth fold
    train_indices, val_indices = list(splits)[
        fold_n
    ]  # get the middle split as test data

    return train_val_data.iloc[val_indices]


if __name__ == "__main__":
    for sub in ["imu0001"]:
        test_model(exp_name="2024-03-19_16-09-59", sub=sub)
