#### label and create windows from strides ####

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as signal
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences

sys.path.append(os.path.join(os.getcwd(), "src"))
from data.DataLoader import DataLoader


class FeatureBuilder:
    """Class to load raw IMU data, label and create windows from strides.
    Always takes only one partition, e.g., only train or only test (because of data normalization).
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        original_data_path: str,
        data_base_path: str,
        config: dict,
        scaler_dict=None,
    ):
        """Initialize FeatureBuilder.

        Args:
            data_df (pd.DataFrame): all IMU data labeled by stride number
            original_data_path (str): path to the original raw data
            data_base_path (str): path to data in this project
            config (dict): configuration dictionary
            scaler ([type], optional): scaler to use for normalizing the data. Defaults to None.
        """
        self.config = config
        self.original_data_path = original_data_path
        self.data_base_path = data_base_path
        self.data_df = data_df
        self.scaler_dict = scaler_dict

        # remove rows with NaN values
        self.data_df = self.data_df.dropna()
        # warning if there are too many NaN values
        if self.data_df.isna().sum().sum() > 10:
            print("****************************************************")
            print(
                f"Warning: there are {self.data_df.isna().sum().sum()} NaN values in the data."
            )
            print("****************************************************")

        # Reset the row indices
        self.data_df = self.data_df.reset_index(drop=True)

        # features are columns that start with "Acc" or "Gyr"
        self.feature_names = self.data_df.columns[
            self.data_df.columns.str.startswith(("Acc", "Gyr"))
        ]

        # smooth the data using a Butterworth filter
        if self.config["smoothing"] == "butterworth":
            self.smooth_data()

    def make_windows(
        self,
        max_window_size: int = None,
        save_windows_name=None,
    ):
        """Create windows from all IMU data labeled by stride number.
        Args:
            max_window_size (int, optional): max window size. Defaults to None.
            save_windows_name ([type], optional): name of the file to save the windowed features and labels. Defaults to None.

        Returns:
            all_windows_ls (list): list of dataframes with each stride as a window
            all_labels_ls (list): list of labels for each stride

        """
        # get unique subs and runs
        subs = self.data_df["sub"].unique()
        runs = self.data_df["run"].unique()

        foot_names = {
            "LF": "left_foot",
            "RF": "right_foot",
        }

        self.all_features_ls = []
        self.all_labels_ls = []
        valid_shifted_gait_events_df_ls = []  # the segmentation foot
        valid_shifted_other_gait_events_df_ls = []  # the non-segmentation foot
        for sub in subs:
            for run in runs:
                valid_idx_ls = []  # list to store valid gait event indices for this run
                valid_other_idx_ls = (
                    []
                )  # list to store valid gait event indices for the other foot
                segmentation_ic1_ls = []  # list to store ic1 for the segmentation foot
                # get data from one subject and one run
                sub_run_df = self.data_df.loc[
                    np.logical_and(
                        self.data_df["sub"] == sub, self.data_df["run"] == run
                    )
                ].reset_index(drop=True)

                # read gait events
                if self.config["dataset"] == "duo_gait":
                    gait_events_dir = os.path.join(
                        os.path.join(self.original_data_path, "processed"), run, sub
                    )
                elif self.config["dataset"] == "charite":
                    gait_events_dir = os.path.join(
                        os.path.join(self.original_data_path, "processed"), sub, run
                    )

                gait_events_df_dict = {}  # dictionary for gait event from both feet
                valid_gait_events_df_dict = (
                    {}
                )  # dictionary for valid gait events from both feet

                for foot in ["segmentation_foot", "other_foot"]:
                    # get gait events for segmentation
                    if foot == "segmentation_foot":
                        gait_events_df_dict[foot] = pd.read_csv(
                            os.path.join(
                                gait_events_dir,
                                f"{foot_names[self.config['segmentation_sensor']]}_core_params.csv",
                            )
                        )
                    # get gait events of the other foot
                    elif foot == "other_foot":
                        gait_events_df_dict[foot] = pd.read_csv(
                            os.path.join(
                                gait_events_dir,
                                f"{foot_names['LF' if self.config['segmentation_sensor'] == 'RF' else 'RF']}_core_params.csv",
                            )
                        )

                    # get only valid strides
                    gait_events_df_dict[foot] = (
                        gait_events_df_dict[foot]
                        .loc[gait_events_df_dict[foot]["is_outlier"] == False]
                        .reset_index(drop=True)
                    )

                    # rename "timestamp" column to "ic_1"
                    if self.config["dataset"] == "duo_gait":
                        gait_events_df_dict[foot].rename(
                            columns={"timestamps": "ic1", "ic_times": "ic2", "fo_times": "fo_time"},
                            inplace=True,
                        )
                    elif self.config["dataset"] == "charite":
                        gait_events_df_dict[foot].rename(
                            columns={"timestamp": "ic1", "ic_time": "ic2"}, inplace=True
                        )

                    # shift gait events relative to the start of each segmentation window (ic1)
                    # for plotting aggregated strides in innvestigate
                    # only for the segmentation foot, align the other foot later for valid strides
                    if foot == "segmentation_foot":
                        gait_events_df_dict[foot + "_shifted"] = gait_events_df_dict[
                            foot
                        ][["ic1", "fo_time", "ic2"]].copy()
                        gait_events_df_dict[foot + "_shifted"] = gait_events_df_dict[
                            foot + "_shifted"
                        ].apply(lambda x: x - x.iloc[0], axis=1)

                # ##### debugging: plot gait events on top of raw data #####
                # plt.figure(figsize=(20, 5))
                # plt.plot(sub_run_df["timestamp"], sub_run_df["GyrX_LF"])
                # plt.vlines(
                #     gait_events_df["ic1"],
                #     ymin=-400,
                #     ymax=650,
                #     colors="r",
                #     label="ic1",
                # )
                # plt.vlines(
                #     gait_events_df["ic2"],
                #     ymin=-450,
                #     ymax=550,
                #     colors="g",
                #     label="ic2",
                # )
                # plt.title(f"Raw data and gait events for {sub} {run}")
                # plt.legend()
                # plt.show()
                # #### end debugging #####

                # cut windows by stride and save valid windows
                for index, row in gait_events_df_dict["segmentation_foot"].iterrows():
                    if np.logical_and(
                        row["ic1"] >= sub_run_df.loc[0, "timestamp"],
                        row["ic2"]
                        <= sub_run_df.loc[sub_run_df.shape[0] - 1, "timestamp"],
                    ):  # if gait event timestamps are in the total range of the data
                        # get data in the current window
                        current_window_df = sub_run_df.loc[
                            np.logical_and(
                                sub_run_df["timestamp"] >= row["ic1"],
                                sub_run_df["timestamp"] < row["ic2"],
                            )
                        ]
                        # if empty window, skip
                        # this could happen if the train test split takes out middle parts of the data
                        if current_window_df.shape[0] == 0:
                            continue

                        # collect features and labels
                        self.all_features_ls.append(
                            current_window_df.loc[:, self.feature_names].values
                        )
                        if self.config["classification_target"] == run:
                            self.all_labels_ls.append(1)
                        else:
                            self.all_labels_ls.append(0)

                        # collect valid gait events
                        valid_idx_ls.append(index)

                        # collect valid gait events from the other foot, if the "fo_time" is within the window
                        other_idx = np.logical_and(
                            gait_events_df_dict["other_foot"]["fo_time"] > row["ic1"],
                            gait_events_df_dict["other_foot"]["fo_time"] < row["ic2"],
                        )
                        if other_idx.sum() == 1:
                            valid_other_idx_ls.append(np.where(other_idx)[0][0])
                            segmentation_ic1_ls.append(row["ic1"])

                # collect shifted gait events from all valid strides
                valid_shifted_gait_events_df_ls.append(
                    gait_events_df_dict["segmentation_foot_shifted"].loc[valid_idx_ls]
                )
                gait_events_df_dict["other_foot_shifted"] = (
                    gait_events_df_dict["other_foot"]
                    .loc[valid_other_idx_ls][["ic1", "fo_time", "ic2"]]
                    .reset_index(drop=True)
                )  # select columns (prepare for shifting) and reset index
                valid_shifted_other_gait_events_df_ls.append(
                    gait_events_df_dict["other_foot_shifted"].subtract(
                        pd.Series(segmentation_ic1_ls), axis=0
                    )
                )

                # compare the number of valid strides in the segmentation and the other foot
                if len(valid_idx_ls) > len(valid_other_idx_ls):
                    # fill rows with NaN values to make the number of valid strides equal
                    n_fill = len(valid_idx_ls) - len(valid_other_idx_ls)
                    valid_shifted_other_gait_events_df_ls[-1] = pd.concat(
                        [
                            valid_shifted_other_gait_events_df_ls[-1],
                            pd.DataFrame(
                                np.nan,
                                index=np.arange(n_fill),
                                columns=valid_shifted_other_gait_events_df_ls[
                                    -1
                                ].columns,
                            ),
                        ],
                        ignore_index=True,
                    )

        # concatenate the valid gait events from all runs into one dataframe
        self.valid_shifted_gait_events_dict = {}
        self.valid_shifted_gait_events_dict["segmentation"] = pd.concat(
            valid_shifted_gait_events_df_ls, ignore_index=True
        )
        self.valid_shifted_gait_events_dict["other"] = pd.concat(
            valid_shifted_other_gait_events_df_ls, ignore_index=True
        )

        # convert labels to numpy arrays
        self.all_labels_ls = np.array(self.all_labels_ls)

        if self.config["pad_windows"]:
            # pad the windows with zeros to make them all the same length
            self.pad_windows(max_window_size=max_window_size)
            self.window_strategy = "padded"
        else:
            # re-sample the windows to 200 time steps, ignore the given max_window_size
            self.all_features_ls = np.array(
                [signal.resample(x, 200, axis=0) for x in self.all_features_ls]
            )
            self.window_strategy = "resampled"

            self.scaling_factor_dict = {}
            for label in np.unique(self.all_labels_ls):
                # get the gait event scaling factor for plotting
                segmentation_ic2 = self.valid_shifted_gait_events_dict[
                    "segmentation"
                ].loc[self.all_labels_ls == label]["ic2"]
                self.scaling_factor_dict[label] = 200 / np.mean(segmentation_ic2)

        # if save windowed features and labels
        if save_windows_name is not None:
            np.savez(
                os.path.join(
                    self.data_base_path,
                    "features",
                    f"windowed_{self.window_strategy}_{save_windows_name}.npz",
                ),
                features=self.all_features_ls,
                labels=self.all_labels_ls,
            )

        return (
            self.all_features_ls,
            self.all_labels_ls,
            self.valid_shifted_gait_events_dict,
        )
    
    def get_gait_event_scaling_factors(self):
        """Get the scaling factors for plotting the gait events."""
        return self.scaling_factor_dict

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        """Apply a low-pass filter to the data.

        Args:
            data (np.array): data to be filtered
            cutoff (float): cutoff frequency
            fs (float): sampling frequency
            order (int, optional): order of the filter. Defaults to 5.

        Returns:
            np.array: filtered data
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq  # normalized cutoff frequency
        b, a = signal.butter(
            order, normal_cutoff, btype="low", analog=False
        )  # get filter coefficients
        y = signal.filtfilt(b, a, data)
        return y

    def smooth_data(self):
        """Smooth the raw IMU data using a Butterworth filter."""
        # smooth the raw IMU data using a Butterworth filter
        for feature in self.feature_names:
            self.data_df.loc[:, feature] = self.butter_lowpass_filter(
                self.data_df.loc[:, feature],
                cutoff=10,
                fs=100,
                order=3,
            )

    def normalize_2d_data(self, plot_title=None):
        """Normalize the raw IMU data using a scaler.

        Args:
            scaler (StandardScaler, optional): scaler. Defaults to None.

        Returns:
            data_2d_scaled (pd.DataFrame): dataframe with normalized IMU data
        """
        # get only the features for normalization
        data_2d = self.data_df.loc[:, self.feature_names]

        if plot_title is not None:
            # plot the first 500 samples before scaling
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
            ax = axs[0]
            for i in range(np.array(data_2d.shape[-1])):
                # plot each feature
                ax.plot(data_2d.iloc[:500, i], label=self.feature_names[i])
            ax.set_title(f"Raw features")
            ax.legend()

        data_2d_scaled_ls = []
        # normalize the features
        if self.scaler_dict is None:
            self.scaler_dict = {}
            for class_label in self.data_df[
                self.config["classification_target"]
            ].unique():
                # get the data for one class
                data_2d_class = self.data_df[
                    self.data_df[self.config["classification_target"]] == class_label
                ]

                # create a new scaler using the data
                self.scaler_dict[class_label] = StandardScaler()
                data_2d_scaled_ls.append(
                    pd.DataFrame(
                        self.scaler_dict[class_label].fit_transform(
                            data_2d_class[self.feature_names].values
                        ),
                        columns=self.feature_names,
                    )
                )
        else:
            for class_label in self.data_df[
                self.config["classification_target"]
            ].unique():
                # get the data for one class
                data_2d_class = self.data_df[
                    self.data_df[self.config["classification_target"]] == class_label
                ]
                # use the given scaler
                data_2d_scaled_ls.append(
                    pd.DataFrame(
                        self.scaler_dict[class_label].transform(
                            data_2d_class[self.feature_names].values
                        ),
                        columns=self.feature_names,
                    )
                )

        # concat dataframe from all the runs
        data_2d_scaled = pd.concat(data_2d_scaled_ls, ignore_index=True)

        if plot_title is not None:
            # plot the first 500 samples after scaling
            ax = axs[1]
            for i in range(np.array(data_2d_scaled.shape[-1])):
                # plot each feature
                ax.plot(data_2d_scaled.iloc[:500, i], label=self.feature_names[i])
            ax.legend()
            ax.set_title(f"Scaled features")
            plt.suptitle(f"Raw and scaled features in {plot_title}")
            plt.tight_layout()

            # save the figure
            save_fig_dir = os.path.join(
                self.data_base_path, "data_exploration", "normalization"
            )
            if not os.path.exists(save_fig_dir):
                os.makedirs(save_fig_dir)
            plt.savefig(os.path.join(save_fig_dir, f"{plot_title}.pdf"))

            # plt.show()
            plt.close()

        # add the scaled features back to the original dataframe
        self.data_df.loc[:, self.feature_names] = data_2d_scaled

    def get_scaler(self):
        """Get the scaler from data normalization."""
        return self.scaler_dict

    def pad_windows(self, max_window_size: int):
        """Pad the windows with zeros to make them all the same length.

        Args:
            max_window_size (int): max window size
        """

        # pad the windows with zeros to make them all the same length
        if (
            max_window_size is None
        ):  # if not specified, use the max window size in the data
            max_window_size = max([len(x) for x in self.all_features_ls])

        self.all_features_ls = pad_sequences(
            self.all_features_ls,
            padding="post",
            truncating="post",
            maxlen=max_window_size,
            dtype="float32",
        )

    def get_feature_names(self):
        """Get feature names as a list."""
        feature_names = list(
            self.data_df.columns[self.data_df.columns.str.startswith(("Acc", "Gyr"))]
        )

        return feature_names

    def plot_features_window(self, window_index: int):
        """Plot all features for quality control.

        Args:
            window_index (int): index of the window to plot
        """

        # plot features
        plt.figure(figsize=(20, 5))
        for i in range(np.array(self.all_features_ls[0]).shape[-1]):
            # plot each feature
            plt.plot(
                self.all_features_ls[window_index][:, i],
                label=self.get_feature_names()[i],
            )
        plt.title(f"Features in window {window_index}")
        plt.ylabel("Sample index")
        plt.legend()
        # plt.show()

    def plot_all_windows(self, sensors: list, sub: str, save_fig_path: str = None):
        """Plot all windows for all features of one sensor, colored by label.

        Args:
            sensors (list): list of sensor names
            sub (str): subject name
        """
        for sensor in sensors:  # make a figure for each sensor
            # plot all windows for all features of one sensor, coloed by label
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
            plt.subplots_adjust(hspace=0.4)  # increase space for subplot titles

            for ax, feature in zip(
                axs.ravel(), ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"]
            ):
                for i in range(len(self.all_features_ls)):
                    ax.plot(
                        self.all_features_ls[i][
                            :, self.get_feature_names().index(f"{feature}_{sensor}")
                        ],
                        alpha=0.1,
                        color="b" if self.all_labels_ls[i] == 1 else "r",
                    )
                # plot average for each label with line width 2
                ax.plot(
                    np.mean(
                        self.all_features_ls[self.all_labels_ls == 1][
                            :, :, self.get_feature_names().index(f"{feature}_{sensor}")
                        ],
                        axis=0,
                    ),
                    color="b",
                    linewidth=2,
                    label="visit2",
                )
                ax.plot(
                    np.mean(
                        self.all_features_ls[self.all_labels_ls == 0][
                            :, :, self.get_feature_names().index(f"{feature}_{sensor}")
                        ],
                        axis=0,
                    ),
                    color="r",
                    linewidth=2,
                    label="visit1",
                )

                # plot average gait events
                if self.config["dataset"] == "duo_gait":
                    sampling_rate = 128
                elif self.config["dataset"] == "charite":
                    sampling_rate = 120

                for label in [0, 1]:
                    if label == 0:
                        line_color = "r"
                        line_style = "--"
                    elif label == 1:
                        line_color = "b"
                        line_style = ":"

                    if label in self.all_labels_ls:
                        # if labeled class is present (sometimes only one class is present in a fold)
                        for gait_event in ["ic1", "fo_time", "ic2"]:
                            if np.logical_and(
                                sensor in ["LF", "RF"],
                                sensor != self.config["segmentation_sensor"],
                            ):
                                # if it is the other foot, use the other foot's gait events
                                side = "other"
                            else:
                                # use segmentation foot's gait events for all other cases
                                side = "segmentation"

                            # get the mean gait event sample for plotting
                            mean_gait_event_time = np.mean(
                                self.valid_shifted_gait_events_dict[side].loc[
                                    self.all_labels_ls == label
                                ][gait_event]
                            )
                            if self.config["pad_windows"]:
                                mean_gait_event_sample = int(
                                    mean_gait_event_time * sampling_rate
                                )
                            else:
                                # in resampled windows, gait events should also be "resampled"
                                mean_gait_event_sample = int(
                                    mean_gait_event_time * self.scaling_factor_dict[label]
                                )

                            if mean_gait_event_time >= 0:
                                # only plot if the gait event is in the data
                                ax.axvline(
                                    mean_gait_event_sample,
                                    color=line_color,
                                    linestyle=line_style,
                                    # label=gait_event,
                                )
                                # add text for label above the lines
                                if label == 0:
                                    ax.text(
                                        mean_gait_event_sample,
                                        ax.get_ylim()[1]
                                        + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                                        gait_event,
                                        color="gray",
                                        fontsize=12,
                                        ha="center",  # horizontal alignment
                                    )

                ax.set_title(feature, y=1.05)
                ax.set_xlabel("Sample index")
                ax.legend()

            
            if self.config["dataset"] == "charite":
                plt.suptitle(
                    f"{sensor} in all windows {sub}, paretic side = {self.config['paretic_side']}\n"
                    + f" n_visit1 = {sum(self.all_labels_ls == 0)}, "
                    + f"n_visit2 = {sum(self.all_labels_ls == 1)}",
                    # y=1.01,
                )
            elif self.config["dataset"] == "duo_gait":
                plt.suptitle(
                    f"{sensor} in all windows {sub}\n"
                    + f" n_visit1 = {sum(self.all_labels_ls == 0)}, "
                    + f"n_visit2 = {sum(self.all_labels_ls == 1)}",
                    # y=1.01,
                )
            plt.tight_layout()
            # plt.subplots_adjust(top=0.88)  # Adjust the top space so the suptitle fits into the figure

            if save_fig_path is not None:
                if not os.path.exists(save_fig_path):
                    os.makedirs(save_fig_path)
                plt.savefig(
                    os.path.join(
                        save_fig_path,
                        f"windowed_features_{self.window_strategy}_{sub}_{sensor}.pdf",
                    )
                )

            # plt.show()
            plt.close(fig)


if __name__ == "__main__":
    subs = [
        "imu0001",
        # "imu0002",
        # "imu0003",
        # "imu0006",
        # "imu0007",
        # "imu0008",
        # "imu0009",
        # "imu0011",
        # "imu0012",
        # "imu0013",
    ]

    config = {}
    config["sensors"] = ["LF"]  # , "RF", "SA"
    config["segmentation_sensor"] = (
        "LF"  # sensor used for stride segmentation using initial contact
    )
    config["runs"] = ["visit1", "visit2"]
    config["classification_target"] = "visit2"  # one of the runs
    config["stride_segmentation"] = True  # segment data by strides

    # get raw IMU data
    with open("path.json") as f:
        paths = json.load(f)
    charite_base_path = paths["data_charite_original"]
    data_base_path = paths["data_charite"]

    # load all data
    data_loader = DataLoader(
        original_data_path=charite_base_path,
        data_base_path=data_base_path,
        subs=subs,
        config=config,
    )
    all_data = data_loader.get_all_data()

    feature_builder = FeatureBuilder(
        all_data, charite_base_path, data_base_path, config
    )
    feature_builder.make_windows()
    feature_builder.plot_features_window(window_index=4)
