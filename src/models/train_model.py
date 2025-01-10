import os
import pickle
import numpy as np
import pandas as pd
import itertools
from matplotlib import pyplot as plt

from features.FeatureBuilder import FeatureBuilder

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from keras import models, layers
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical

# import wandb
import yaml
# from wandb.keras import WandbCallback


def get_CNN_model(config: dict, window_size: int, n_features: int):
    """Create a CNN model

    Args:
        cfg (dict): configuration dictionary
        win_size (int): window size.
        n_features (int): number of features.

    Returns:
        model: the model
    """
    if config["use_wandb"]:
        n_filters = wandb.config.num_filters
        n_layers = wandb.config.num_layers
        lr = wandb.config.learning_rate
    else:
        n_filters = config["n_filters"]
        n_layers = config["n_layers"]
        lr = config["initial_lr"]

    model = models.Sequential()
    model.add(
        layers.Conv1D(
            filters=n_filters,
            kernel_size=5,
            activation="relu",
            input_shape=(window_size, n_features),
        )
    )  # (10, 6) kernel_size=5 filters=32
    # Add additional convolutional layers based on the num_layers hyperparameter
    for i in range(
        n_layers - 1,
    ):
        model.add(
            layers.Conv1D(
                filters=n_filters,
                kernel_size=5,
                activation="relu",
            )
        )
    model.add(layers.GlobalMaxPooling1D())  # or model.add(Flatten())
    model.add(layers.Dense(2, activation="softmax"))  # sigmoid
    print(model.summary())
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=lr),  # "binary_crossentropy"
        metrics=[config["evaluation_metric"]],
    )  # , precision, recall gives an error with the combined versions of keras-metrics, keras, and tf
    return model


def execute_training(X_train, y_train, X_val, y_val, config, save_path):
    """Train model (deprecated, this one does not optimize hyperparameters, use train_model() instead)

    Args:
        X_train (_type_): train data
        y_train (_type_): train labels
        X_val (_type_): validation data
        y_val (_type_): validation labels
        cfg (dict): configuration dictionary

    Returns:
        _type_: _description_
    """
    # get sweep config
    with open(os.path.join(".", "src", "models", "sweep_config.yaml")) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
    # copy the sweep config to experiment folder
    with open(os.path.join(save_path, "sweep_config.yaml"), "w") as file:
        yaml.dump(sweep_config, file)
    if config["use_wandb"]:
        # initialize wandb
        wandb.init(project=config["experiment_name"], config=sweep_config)

    # callbacks
    callbacks = []
    if config["use_wandb"]:
        callbacks.append(
            WandbCallback()
        )  # automatically saves the model and logs metrics with wandb
    if config["early_stopping"]:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3))
    if config["reduce_lr"]:
        callbacks.append(
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=0.001)
        )

    # get the model
    model = get_CNN_model(
        config=config,
        window_size=X_train.shape[1],
        n_features=X_train.shape[2],
    )

    # two output neurons (for softmax) compatability
    if model.output.shape.ndims == 2:
        # reshape target labels
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )  # , validation_split=0.2
    y_pred = np.argmax(model.predict(X_val), axis=-1)

    if len(y_val.shape) == 1:
        y_val = y_val[:, None]
    elif len(y_val.shape) == 2:  # if two dimensions (i.e., one-hot coded label)
        # undo the one-hot encoding
        y_val = np.argmax(y_val, axis=1)
    confusion_mat = confusion_matrix(y_true=y_val, y_pred=y_pred)
    print("train val confusion matrix:\n", confusion_mat)

    # plot results
    plot_results(
        history, metrics=[config["evaluation_metric"]], save_path=save_path, show=False
    )

    # save model
    model.save(os.path.join(save_path, "model.h5"))

    return history, model


def train_model(
    config: dict,
    train_val_data,
    original_data_path,
    data_base_path,
    data_exploration_dir,
    sub,
):
    """Train model with grid search and
    use average score from 5-fold cross-validation as metric
    to select the best model hyperparameters

    Args:
        cfg (dict): configuration dictionary
        train_val_data (_type_): train and validation data
        original_data_path (_type_): path to the original data
        data_base_path (_type_): path to the base data
        data_exploration_dir (_type_): path to the data exploration directory
        sub (str): subject name

    Returns:
        _type_: _description_
    """

    # Get all combinations of hyperparameters
    # read the grid from the yaml file
    with open(
        os.path.join(
            data_base_path,
            "results",
            config["experiment_name"],
            "hyperparameter_grid.yaml",
        )
    ) as file:
        grid = yaml.load(file, Loader=yaml.FullLoader)
    combinations = list(itertools.product(*grid.values()))

    ############################
    # hyperparameter optimization with 5-fold cross-validation and grid search

    # Loop through all combinations of hyperparameters
    best_model = None
    best_params = None
    best_score = 0
    for combination in combinations:
        params = dict(zip(grid.keys(), combination))
        config.update(params)  # update the config with the current hyperparameters

        # split the data into training and validation sets
        skf_train_val = StratifiedKFold(
            5, shuffle=False
        )  # 5-fold cross-validation for train/validation split

        fold_num = 1
        metric_per_fold = []
        for i, (train_index, val_index) in enumerate(
            skf_train_val.split(
                train_val_data, train_val_data[config["classification_target"]]
            )
        ):
            # reshape the data into windows
            # get raw train and val data (time series before windowing)
            X_y_train_raw = train_val_data.iloc[train_index]
            X_y_val_raw = train_val_data.iloc[val_index]

            # get stride windows for train data
            feature_builder_train = FeatureBuilder(
                original_data_path=original_data_path,
                data_base_path=data_base_path,
                data_df=X_y_train_raw,
                config=config,
                scaler_dict=None,  # create new scalers for training data
            )
            feature_builder_train.normalize_2d_data(plot_title=f"{sub} train data")
            train_scaler_dict = feature_builder_train.get_scaler()
            X_train, y_train, _ = feature_builder_train.make_windows()
            feature_builder_train.plot_all_windows(
                sensors=config["sensors"],
                sub=sub,
                save_fig_path=os.path.join(data_exploration_dir, "train_windows"),
            )

            # get stride windows for validation data
            feature_builder_val = FeatureBuilder(
                original_data_path=original_data_path,
                data_base_path=data_base_path,
                data_df=X_y_val_raw,
                config=config,
                scaler_dict=train_scaler_dict,  # use the same scaler as the training data
            )
            feature_builder_val.normalize_2d_data(plot_title=f"{sub} val data")
            X_val, y_val, _ = feature_builder_val.make_windows(
                max_window_size=X_train.shape[1]
            )  # use the same window size as the training data
            feature_builder_val.plot_all_windows(
                sensors=config["sensors"],
                sub=sub,
                save_fig_path=os.path.join(data_exploration_dir, "val_windows"),
            )

            # save the scaler
            scaler_path = os.path.join(
                data_base_path,
                "results",
                config["experiment_name"],
                sub,
                f"scaler_fold_{fold_num}.pkl",
            )
            with open(scaler_path, "wb") as f:
                pickle.dump(train_scaler_dict, f)

            # get the model
            model = get_CNN_model(
                config=config,
                window_size=X_train.shape[1],
                n_features=X_train.shape[2],
            )

            # Generate a print
            print(
                "\n\n\n------------------------------------------------------------------------"
            )
            print(f"Training for fold {fold_num} ...")
            for y_name, y in zip(["y_train", "y_val"], [y_train, y_val]):
                print(f"{y_name} class distribution: {np.unique(y, return_counts=True)}")

            # callbacks
            callbacks = []
            if config["use_wandb"]:
                callbacks.append(
                    WandbCallback()
                )  # automatically saves the model and logs metrics with wandb
            if config["early_stopping"]:
                callbacks.append(EarlyStopping(monitor="val_loss", patience=3))
            if config["reduce_lr"]:
                callbacks.append(
                    ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=2, min_lr=0.001
                    )
                )

            # two output neurons (for softmax) compatability
            if model.output.shape.ndims == 2:
                # reshape target labels
                y_train = to_categorical(y_train, num_classes=2)
                y_val = to_categorical(y_val, num_classes=2)

            history = model.fit(
                X_train,
                y_train,
                epochs=25,
                batch_size=config["batch_size"],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )  # , validation_split=0.2
            y_pred = np.argmax(model.predict(X_val), axis=-1)

            # if len(y_val.shape) == 1:
            #     y_val = y_val[:, None]
            # elif len(y_val.shape) == 2:  # if two dimensions (i.e., one-hot coded label)
            #     # undo the one-hot encoding
            #     y_val = np.argmax(y_val, axis=1)

            # plot results
            save_plot_path = os.path.join(
                data_base_path,
                "results",
                config["experiment_name"],
                sub,
                f"fold_{fold_num}",
            )
            os.makedirs(save_plot_path, exist_ok=True)
            plot_results(
                history,
                metrics=[config["evaluation_metric"]],
                save_path=save_plot_path,
                show=False,
            )

            # Generate generalization metrics
            scores = model.evaluate(X_val, y_val, verbose=0)
            print(
                f"Score for fold {fold_num}: "
                + f"{model.metrics_names[0]} of {scores[0]}; "
                + f"{model.metrics_names[1]} of {scores[1]}"
            )
            metric_per_fold.append(scores[1] * 100)

            # Increase fold number
            fold_num = fold_num + 1

        # print the average scores
        print(
            "------------------------------------------------------------------------"
        )
        print("Score per fold \n")
        for i in range(0, len(metric_per_fold)):
            print(
                f"> Fold {i+1} - {config['evaluation_metric']}: {metric_per_fold[i]} \n"
            )
        print(
            "------------------------------------------------------------------------"
        )
        print("Average scores for all folds:")
        print(
            f"> {config['evaluation_metric']}: {np.mean(metric_per_fold)} (+- {np.std(metric_per_fold)})"
        )
        print(
            "------------------------------------------------------------------------"
        )

        if np.mean(metric_per_fold) > best_score:
            best_score = np.mean(metric_per_fold)
            best_params = params
            best_model = model

    # save the best hyperparameters
    with open(
        os.path.join(
            data_base_path,
            "results",
            config["experiment_name"],
            sub,
            "best_params.yaml",
        ),
        "w",
    ) as file:
        yaml.dump(best_params, file)

    ############################
    # re-train the models with the best hyperparameters on
    # the entire training and validation data and save the best model
    print("\n\n\n=====================================================================")
    print("Training the best model on the entire training and validation data ...")

    # make windows for the entire training and validation data
    feature_builder_train_val = FeatureBuilder(
        original_data_path=original_data_path,
        data_base_path=data_base_path,
        data_df=train_val_data,  # all train and validation data
        config=config,
        scaler_dict=None,  # create new scaler for training data
    )
    feature_builder_train_val.normalize_2d_data(plot_title=f"{sub} train data")
    train_val_scaler_dict = feature_builder_train_val.get_scaler()
    X_train_val, y_train_val, _ = feature_builder_train_val.make_windows()
    feature_builder_train_val.plot_all_windows(
        sensors=config["sensors"],
        sub=sub,
        save_fig_path=os.path.join(data_exploration_dir, "train_val_windows"),
    )

    # get the best model
    config.update(best_params)  # update the config with the best hyperparameters
    best_hyperparam_model = get_CNN_model(
        config=config,
        window_size=X_train_val.shape[1],
        n_features=X_train_val.shape[2],
    )

    # callbacks
    callbacks = []
    if config["use_wandb"]:
        callbacks.append(
            WandbCallback()
        )  # automatically saves the model and logs metrics with wandb
    if config["early_stopping"]:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=3))
    if config["reduce_lr"]:
        callbacks.append(
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=0.001
            )
        )

    # two output neurons (for softmax) compatability
    if best_hyperparam_model.output.shape.ndims == 2:
        # reshape target labels
        y_train_val = to_categorical(y_train_val, num_classes=2)

    # Use a portion of the training data as a pseudo-validation set for early stopping and learning rate reduction
    validation_split = 0.1

    best_hyperparam_model.fit(
        X_train_val,
        y_train_val,
        epochs=25,
        batch_size=config["batch_size"],
        callbacks=callbacks,
        validation_split=validation_split,
    )

    # save the train_val scaler to use on the test data
    train_val_scaler_path = os.path.join(
        data_base_path,
        "results",
        config["experiment_name"],
        sub,
        "train_val_scaler.pkl",
    )
    with open(train_val_scaler_path, "wb") as f:
        pickle.dump(train_val_scaler_dict, f)

    # save the best model
    best_hyperparam_model.save(
        os.path.join(
            data_base_path, "results", config["experiment_name"], sub, "best_model.h5"
        )
    )

def plot_results(history, metrics: list, save_path=None, show=False):
    """
    Plots loss and performance over epochs.

    Parameters
    ----------
    history: the history object that is returned by the model.fit() function
    metrics: a list of metrics of the history that should be plotted
    save: True if plots and metrics should be saved

    Returns: None
    -------

    """
    history_dict = history.history

    if save_path:
        result_df = pd.DataFrame.from_dict(history_dict).round(3)
        result_df.to_csv(os.path.join(save_path, "results.csv"), index=False)
        plot_path = os.path.join(save_path, "plots")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)  # the plot folder

    metrics.append("loss")
    for metric in metrics:
        if metric == "AUC":
            metric = "auc"  # convert to lowercase
        train_values = history_dict[metric]
        val_metric = "val_" + metric
        val_values = history_dict[val_metric]
        epochs = range(1, len(train_values) + 1)

        fig = plt.figure()
        plt.plot(epochs, train_values, "bo", label=f"Training {metric}")
        plt.plot(epochs, val_values, "b", label=f"Validation {val_metric}")
        plt.title(f"Training and validation {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(f"{metric}")
        plt.legend()
        if save_path:
            fpath = os.path.join(plot_path, f"{metric}.pdf")
            plt.savefig(fpath)
        if show:
            plt.show()
        plt.clf()
        plt.close(fig)


def normalize_3d_data(
    arr_train: np.ndarray,
    arr_val: np.ndarray,
    scaler=None,
    plot_windows_title=None,
    save_fig_path=None,
):
    """Normalize 3d data

    Args:
        arr_train (np.ndarray): 3d data used for fitting the scaler
        arr_val (np.ndarray): 3d data to be normalized using the scaler
        scaler (StandardScaler): the scaler to be used for normalization, if None, a new scaler is created
        plot_windows_title (str, optional): whether to plot the first window before and after scaling. Defaults to None.
        save_fig_path (str, optional): the base path to save the plots. Defaults to None.

    Returns:
        arr_train_scaled (np.ndarray): normalized training data
        arr_val_scaled (np.ndarray): normalized validation data
        scaler (StandardScaler): the scaler used for normalization

    """
    if plot_windows_title is not None:
        # plot the first 5 windows before scaling
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))
        for i in range(5):
            ax = axs[0, i]
            for j in range(arr_train.shape[-1]):
                ax.plot(arr_train[i, :, j])
            ax.set_title(f"window {i} before scaling")

    # temporarily reshape the data to fit the scaler
    arr_train_2d = arr_train.reshape(-1, arr_train.shape[-1])
    arr_val_2d = arr_val.reshape(-1, arr_val.shape[-1])

    # fit the scaler
    if scaler is None:
        scaler = StandardScaler()
        arr_train_scaled_2d = scaler.fit_transform(arr_train_2d)
        arr_val_scaled_2d = scaler.transform(arr_val_2d)
    else:
        # use the scaler from the training data
        arr_train_scaled_2d = scaler.transform(arr_train_2d)
        arr_val_scaled_2d = scaler.transform(arr_val_2d)

    # reshape back to 3d
    arr_train_scaled = arr_train_scaled_2d.reshape(arr_train.shape)
    arr_val_scaled = arr_val_scaled_2d.reshape(arr_val.shape)

    if plot_windows_title is not None:
        # plot the first window after scaling
        for i in range(5):
            ax = axs[1, i]
            for j in range(arr_train_scaled.shape[-1]):
                ax.plot(arr_train_scaled[i, :, j])
            ax.set_title(f"window {i} after scaling")
        plt.suptitle(plot_windows_title)
        # create the plots folder if it does not exist
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        plt.savefig(os.path.join(save_fig_path, f"{plot_windows_title}.pdf"))
        # plt.show()

    return arr_train_scaled, arr_val_scaled, scaler
