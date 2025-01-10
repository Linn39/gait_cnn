#### load IMU data ####

import os
import json
import pandas as pd

with open("path.json", "r") as f:
    paths = json.load(f)
data_base_path = paths["data_charite"]


class DataLoader:
    """Class to load and label raw IMU data."""

    def __init__(
        self, original_data_path: str, data_base_path: str, subs: list, config: dict
    ):
        """Initialize DataLoader.

        Args:
            original_data_path (str): path to the original raw data
            data_base_path (str): path to data in this project
            subs (list): list of subject IDs
            config (dict): configuration dictionary
        """
        self.config = config
        self.original_data_path = original_data_path
        self.data_base_path = data_base_path
        self.subs = subs

    def load_raw_data(self):
        """Load raw IMU data from original dataset.

        Args:
            subs (list): list of subject IDs

        Returns:
            dict: dictionary with raw IMU data
        """

        # check if the sensor for segmentation is in the list of sensors
        if self.config["segmentation_sensor"] not in self.config["sensors"]:
            raise ValueError(
                f"The sensor **{self.config['segmentation_sensor']}** for segmentation"
                + " must be in the list of sensors."
            )

        # create an empty list to store data
        data_list = []
        for sub in self.subs:
            for run in self.config["runs"]:
                all_sensors_list = []
                for sensor in self.config["sensors"]:
                    # read data
                    if self.config["dataset"] == "duo_gait":
                        data_df = pd.read_csv(
                            os.path.join(
                                self.original_data_path,
                                "interim",
                                run,
                                sub,
                                f"{sensor}.csv",
                            )
                        )
                    elif self.config["dataset"] == "charite":
                        data_df = pd.read_csv(
                            os.path.join(
                                self.original_data_path,
                                "interim",
                                sub,
                                run,
                                "imu",
                                f"{sensor}.csv",
                            )
                        )

                    # drop unnecessary columns
                    data_df.drop(columns=["Unnamed: 0"], inplace=True)

                    # add sensor name as suffix to column names, except for time
                    data_df.columns = [
                        f"{col}_{sensor}" if col != "timestamp" else col
                        for col in data_df.columns
                    ]
                    all_sensors_list.append(data_df)

                # merge all sensors into one dataframe by the timestamp column
                all_sensors_df = self.concat_sensors(all_sensors_list)

                # reset timestamp to start from 0
                all_sensors_df["timestamp"] = (
                    all_sensors_df["timestamp"] - all_sensors_df["timestamp"][0]
                )

                # label data
                all_sensors_df["sub"] = sub
                all_sensors_df["run"] = run
                if self.config["classification_target"] == run:
                    all_sensors_df[self.config["classification_target"]] = 1
                else:
                    all_sensors_df[self.config["classification_target"]] = 0

                data_list.append(all_sensors_df)

        self.all_data = pd.concat(data_list, axis=0)

    def concat_sensors(self, sensors: list):
        """Concat IMU data from different sensors, align the timestamps.

        Args:
            sensors (list): list of dataframes from different sensors

        Returns:
            dataframe: dataframe with all sensors and one common timestamp column
        """

        # merge one sensor at a time using the timestamp column
        all_sensors_df = sensors[0]
        for sensor in sensors[1:]:
            all_sensors_df = pd.merge(
                all_sensors_df, sensor, how="outer", on="timestamp"
            )
            if len(all_sensors_df) != len(sensor):
                print(
                    f"Dropped {len(sensor) - len(all_sensors_df)} rows when merging sensors."
                )

        # sort by timestamp
        all_sensors_df.sort_values(by="timestamp", inplace=True)

        return all_sensors_df

    def get_all_data(self):
        """Return all raw IMU data.

        Returns:
            dataframe: dataframe with all raw IMU data
        """
        # if self.all_data does not exist

        if not hasattr(self, "data"):
            self.load_raw_data()

        return self.all_data
