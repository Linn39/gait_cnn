import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class ResultsPlotter:
    def __init__(self, data_base_path: str, experiment_name: str):
        """Plot the cross-validation and test results.

        Args:
            data_base_path (str): path to the data directory
            experiment_name (str): name of the experiment (LF and RF as features)
            for evaluation metric and for plotting the results

        """
        self.data_base_path = data_base_path
        self.experiment_name = experiment_name

        # read the config
        config = json.load(
            open(
                os.path.join(
                    data_base_path, "results", self.experiment_name, "config.json"
                ),
                "r",
            )
        )
        evaluation_metric = config["evaluation_metric"]
        if evaluation_metric == "AUC":
            self.evaluation_metric = "auc"  # convert to lowercase

    def get_val_test_results(self):
        """Plot and save the cross-validation and test results.

        Args:
            data_base_path (str): path to the data directory
            experiment_name (str): name of the experiment
        """

        # find the subjects in the experiment
        sub_list = [
            sub
            for sub in os.listdir(
                os.path.join(data_base_path, "results", self.experiment_name)
            )
            if not sub.startswith(".")
            and os.path.isdir(
                os.path.join(data_base_path, "results", self.experiment_name, sub)
            )
        ]

        # load the validation results for all subs
        self.val_results_df = self.get_train_val_results(self.experiment_name, sub_list)

        # load the test results for all subs
        test_results_path = os.path.join(
            data_base_path, "results", self.experiment_name, "test_results.csv"
        )

        self.test_results_df = pd.read_csv(test_results_path)

        # return val_results_df, test_results_df

    def plot_val_test_results(self):
        """Plot and save the cross-validation and test results."""

        self.get_val_test_results()

        # plot the results
        fig, ax = plt.subplots()

        ax.plot(
            self.val_results_df["sub"],
            self.val_results_df[f"val_{self.evaluation_metric}"],
            "o",
            alpha=0.5,
            label="val",
        )
        ax.plot(
            self.test_results_df["sub"],
            self.test_results_df[self.evaluation_metric],
            "*",
            # alpha=0.5,
            label="test",
        )
        # ax.set_xlabel("Subject")
        ax.set_ylim([0, 1.2])
        ax.set_ylabel(self.evaluation_metric.upper())
        ax.set_title("Cross-validation and test results" + "\n" + self.experiment_name)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                data_base_path, "results", self.experiment_name, "val_test_results.pdf"
            )
        )
        # plt.show()

    def get_paretic_side_results(self, original_data_dir, lf_dir, rf_dir):
        """Collect the paretic side results

        Args:
            original_data_dir (str): path to the original data directory
            lf_dir (str): path to the left paretic side results
            rf_dir (str): path to the right paretic side results
        """

        # get paretic side inforamtion and all subject names
        paretic_side_df = pd.read_csv(
            os.path.join(original_data_dir, "raw", "paretic_side.csv")
        )

        rf_paretic_subs = paretic_side_df.loc[
            paretic_side_df["paretic_side"] == "Right", "sub"
        ].tolist()
        # paretic side for left or non-specified
        lf_paretic_subs = paretic_side_df.loc[
            paretic_side_df["paretic_side"] != "Right", "sub"
        ].tolist()

        # load left and right side test results
        lf_result_df = pd.read_csv(os.path.join(lf_dir, "test_results.csv"))
        rf_result_df = pd.read_csv(os.path.join(rf_dir, "test_results.csv"))

        self.paretic_test_results_df = pd.concat(
            [
                lf_result_df.loc[lf_result_df["sub"].isin(lf_paretic_subs)],
                rf_result_df.loc[rf_result_df["sub"].isin(rf_paretic_subs)],
            ]
        )

        self.non_paretic_test_results_df = pd.concat(
            [
                lf_result_df.loc[lf_result_df["sub"].isin(rf_paretic_subs)],
                rf_result_df.loc[rf_result_df["sub"].isin(lf_paretic_subs)],
            ]
        )

        # load the left and right side corss validation results
        self.paretic_val_results_df = pd.concat(
            [
                self.get_train_val_results(lf_dir, lf_paretic_subs),
                self.get_train_val_results(rf_dir, rf_paretic_subs),
            ]
        )
        self.non_paretic_val_results_df = pd.concat(
            [
                self.get_train_val_results(lf_dir, rf_paretic_subs),
                self.get_train_val_results(rf_dir, lf_paretic_subs),
            ]
        )

        # sort by subject
        for df in [
            self.paretic_test_results_df,
            self.non_paretic_test_results_df,
            self.paretic_val_results_df,
            self.non_paretic_val_results_df,
        ]:
            df.sort_values(by="sub", inplace=True)

    def plot_paretic_side_results(self, original_data_dir, lf_dir, rf_dir):
        """Plot the paretic and non-paretic side results

        Args:
            original_data_dir (str): path to the original data directory
                to get the paretic side information
            lf_dir (str): path to the left paretic side results
            rf_dir (str): path to the right paretic side results
        """

        self.get_paretic_side_results(original_data_dir, lf_dir, rf_dir)

        # plot the results
        fig, ax = plt.subplots()
        ax.plot(
            self.paretic_val_results_df["sub"],
            self.paretic_val_results_df[f"val_{self.evaluation_metric}"],
            "o",
            alpha=0.5,
            label="Validation (Paretic)",
        )
        ax.plot(
            self.paretic_test_results_df["sub"],
            self.paretic_test_results_df[self.evaluation_metric],
            "*",
            label="Test (Paretic)",
        )

        # shift the non-paretic results to the right
        for df in [self.non_paretic_val_results_df, self.non_paretic_test_results_df]:
            # Create a dictionary mapping categories to integers
            categories = df["sub"].unique()
            category_dict = {category: i for i, category in enumerate(categories)}

            # Apply the mapping to the 'sub' column
            df["sub"] = df["sub"].map(category_dict)

        shift_value = 0.2  # adjust the horizontal position
        ax.plot(
            self.non_paretic_val_results_df["sub"] + shift_value,
            self.non_paretic_val_results_df[f"val_{self.evaluation_metric}"],
            "o",
            alpha=0.5,
            label="Validation (Non-Paretic)",
        )
        ax.plot(
            self.non_paretic_test_results_df["sub"] + shift_value,
            self.non_paretic_test_results_df[self.evaluation_metric],
            "*",
            label="Test (Non-Paretic)",
        )

        # ax.set_xlabel("Subject")
        ax.set_ylim([0, 1.2])
        ax.set_ylabel(self.evaluation_metric.upper())
        ax.set_title("Cross-validation and test results" + "\n")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            os.path.join(data_base_path, "results", "paretic_val_test_results.pdf")
        )
        # plt.show()

    def get_train_val_results(self, exp_name, sub_list):
        """Get the training and validation results for all subjects.

        Args:
            experiment_name (str): name of the experiment
            sub_list (list): list of subjects

        Returns:
            pd.DataFrame: training and validation results
        """

        val_results_ls = []
        for sub in sub_list:
            train_val_results = []
            folds = []
            for fold in os.listdir(
                os.path.join(data_base_path, "results", exp_name, sub)
            ):
                if fold.startswith("fold"):  # only select fold directories
                    # load the validation results
                    val_results_path = os.path.join(
                        data_base_path, "results", exp_name, sub, fold, "results.csv"
                    )
                    sub_val_acc_df = pd.read_csv(val_results_path)
                    val_acc = sub_val_acc_df.loc[sub_val_acc_df.index[-1], "val_auc"]
                    train_val_results.append(val_acc)
                    folds.append(fold)

            # combine the lists into a dataframe
            val_results_df = pd.DataFrame(
                {"fold": folds, "val_auc": train_val_results, "sub": sub}
            )
            val_results_ls.append(val_results_df)

        # combine the dataframes into one
        all_val_results_df = pd.concat(val_results_ls)
        all_val_results_df = all_val_results_df.sort_values(by="sub")  # sort by subject

        return all_val_results_df

    def plot_all_with_paretic_side(self, original_data_dir, lf_dir, rf_dir):
        """Plot the results for both feet and the paretic side."""

        self.get_val_test_results()
        self.get_paretic_side_results(original_data_dir, lf_dir, rf_dir)

        # plot the results
        fig, ax = plt.subplots()
        ax.plot(
            self.val_results_df["sub"],
            self.val_results_df[f"val_{self.evaluation_metric}"],
            "o",
            alpha=0.5,
            label="Validation (Both Sides)",
        )
        ax.plot(
            self.test_results_df["sub"],
            self.test_results_df[self.evaluation_metric],
            "*",
            label="Test (Both Sides)",
        )

        # shift the non-paretic results to the right
        for df in [
            self.paretic_val_results_df,
            self.paretic_test_results_df,
            self.non_paretic_val_results_df,
            self.non_paretic_test_results_df,
        ]:
            # Create a dictionary mapping categories to integers
            categories = df["sub"].unique()
            category_dict = {category: i for i, category in enumerate(categories)}

            # Apply the mapping to the 'sub' column
            df["sub"] = df["sub"].map(category_dict)

        shift_paretic = 0.2
        ax.plot(
            self.paretic_val_results_df["sub"] + shift_paretic,
            self.paretic_val_results_df[f"val_{self.evaluation_metric}"],
            "o",
            alpha=0.5,
            label="Validation (Paretic)",
        )
        ax.plot(
            self.paretic_test_results_df["sub"] + shift_paretic,
            self.paretic_test_results_df[self.evaluation_metric],
            "*",
            label="Test (Paretic)",
        )

        shift_non_paretic = 0.4
        ax.plot(
            self.non_paretic_val_results_df["sub"] + shift_non_paretic,
            self.non_paretic_val_results_df[f"val_{self.evaluation_metric}"],
            "o",
            alpha=0.5,
            label="Validation (Non-Paretic)",
        )
        ax.plot(
            self.non_paretic_test_results_df["sub"] + shift_non_paretic,
            self.non_paretic_test_results_df[self.evaluation_metric],
            "*",
            label="Test (Non-Paretic)",
        )

        ax.set_ylim([0, 1.2])
        ax.set_ylabel(self.evaluation_metric.upper())
        ax.set_title("Cross-validation and test results" + "\n")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            os.path.join(data_base_path, "results", "both_paretic_val_test_results.pdf")
        )

    def get_average_scores(self):
        """Get the average scores for all subjects,
        both feet, paretic side and non-paretic side.

        Args:
            results_df (pd.DataFrame): results dataframe

        Returns:
            pd.DataFrame: average results
        """

        self.get_val_test_results()
        self.get_paretic_side_results(original_data_dir, lf_dir, rf_dir)

        # get the average test scores for both feet
        both_feet_avg = self.test_results_df[self.evaluation_metric].mean()
        both_feet_std = self.test_results_df[self.evaluation_metric].std()

        # get the average scores for the paretic side
        paretic_avg = self.paretic_test_results_df[self.evaluation_metric].mean()
        paretic_std = self.paretic_test_results_df[self.evaluation_metric].std()

        # get the average scores for the non-paretic side
        non_paretic_avg = self.non_paretic_test_results_df[
            self.evaluation_metric
        ].mean()
        non_paretic_std = self.non_paretic_test_results_df[self.evaluation_metric].std()

        print(
            "Test scores for all subjects\n"
            + f"Both feet: {both_feet_avg:.2f} ± {both_feet_std:.2f}"
            + "\n"
            + f"Paretic side: {paretic_avg:.2f} ± {paretic_std:.2f}"
            + "\n"
            + f"Non-paretic side: {non_paretic_avg:.2f} ± {non_paretic_std:.2f}"
        )
        print()
        print("Test scores from paretic side data:")
        print(self.paretic_test_results_df)
        print()
        print("Test scores from non-paretic side data:")
        print(self.non_paretic_test_results_df)


# main
if __name__ == "__main__":

    with open("path.json", "r") as f:
        paths = json.load(f)
    original_data_dir = paths["data_charite_original"]
    data_base_path = paths["data_charite"]

    result_plotter = ResultsPlotter(
        data_base_path, "2024-04-12_17-09-47_LF_RF_resample"
    )

    # plot cross-validation and test results
    result_plotter.plot_val_test_results()

    # plot results by paretic side
    lf_dir = os.path.join(data_base_path, "results", "2024-04-12_16-58-47_LF_resample")
    rf_dir = os.path.join(data_base_path, "results", "2024-04-12_17-41-35_RF_resample")
    result_plotter.plot_paretic_side_results(original_data_dir, lf_dir, rf_dir)

    # plot results by both feet and the paretic side
    result_plotter.plot_all_with_paretic_side(original_data_dir, lf_dir, rf_dir)

    # get the average scores
    result_plotter.get_average_scores()
