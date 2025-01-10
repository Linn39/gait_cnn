#### innvestigate a model
import os
import json
import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.collections import LineCollection
import seaborn as sns

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import innvestigate

from features.FeatureBuilder import FeatureBuilder


def plot_overall_feature_relevance_for_sub(
    analysis, features, plt_info, split_path, subject
):
    """Plots the feature relevance bar plots for all samples of a given subject."""
    # build path
    split_path = os.path.join(
        split_path, "innvestigate_plots", "complete_feature_relevance_plots"
    )
    if not os.path.exists(split_path):
        os.makedirs(split_path)

    # get respective indices and plot each sample's feature relevance
    pos_indices = plt_info[
        ((plt_info["sub"] == subject) & (plt_info["y_true"] == 1))
    ].index
    neg_indices = plt_info[
        ((plt_info["sub"] == subject) & (plt_info["y_true"] == 0))
    ].index
    for idx in pos_indices:
        relevance_sums = np.sum(
            analysis[idx, :, :], axis=0
        )  # relevance sum for each feature
        idx_str = format(idx, "06d")  # to convert e.g. 911 to "000911"
        plot_overall_feature_relevance(
            relevance_sums,
            features,
            os.path.join(
                split_path, f"{subject}_pos_{idx_str}_overall_feature_relevance"
            ),
            plt_info.iloc[idx],
            idx,
        )
    for idx in neg_indices:
        relevance_sums = np.sum(
            analysis[idx, :, :], axis=0
        )  # relevance sum for each feature
        idx_str = format(idx, "06d")
        plot_overall_feature_relevance(
            relevance_sums,
            features,
            os.path.join(
                split_path, f"{subject}_neg_{idx_str}_overall_feature_relevance"
            ),
            plt_info.iloc[idx],
            idx,
        )


def plot_overall_feature_relevance(
    relevance_sums, features, save_path, plt_info, sample_idx, avg_lines=True
):
    """
    Plots a horizontal barplot of the relevance sum of each feature.

    Parameters
    ----------
    relevance_sums
    features
    save_path
    plt_info
    sample_idx

    Returns
    -------

    """
    # adjust font sizes to paper scaling
    font_size = 20
    params = {
        "legend.fontsize": font_size,  # 'x-large'
        # 'figure.figsize': (15, 5),
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    }
    pylab.rcParams.update(params)

    if dataset == "charite":
        paretic_side_text = f"Paretic Side:"
    else:
        paretic_side_text = ""
    if type(plt_info) == pd.DataFrame:  # Multiple strides (aggr. scenario)
        sub = plt_info["sub"].iloc[0]
        y_true = plt_info["y_true"].iloc[0]
        title = (
            f"Summed LRP relevances ({sub}, y_true = {y_true}, sample {sample_idx})\n"
            + f"{paretic_side_text} {plt_info['paretic_side'].iloc[0]}"
        )
    else:  # pd.Series
        sub = plt_info["sub"]
        y_true = plt_info["y_true"]
        y_pred = plt_info["y_pred"]
        title = (
            f"Summed LRP relevances ({sub}, y_true = {y_true}, y_pred = {y_pred}, sample {sample_idx})\n"
            + f"{paretic_side_text} {plt_info['paretic_side']}"
        )

    fig, ax = plt.subplots(figsize=(7, 8))
    # fig.suptitle(title)

    dat = pd.DataFrame({"feature": features, "relevance": relevance_sums})
    dat["pos_relevance"] = dat["relevance"] > 0
    color_map = {True: "#ef8a62", False: "#67a9cf"}
    sns.barplot(
        data=dat,
        x="relevance",
        y="feature",
        hue="pos_relevance",
        palette=color_map,
        dodge=False,
    )
    ax.set(
        ylabel="Features", xlabel="Summed Relevance"
    )  # xlim=(int(np.min(relevance_sums)) - 1, int(np.max(relevance_sums)) + 1),
    sns.despine(left=True, bottom=True)
    plt.legend([], [], frameon=False)

    dat_wo_statics = dat.iloc[
        :-4
    ]  # disregard the static relevances as they distort the mean
    if avg_lines:
        plt.axvline(
            x=dat_wo_statics[dat_wo_statics["relevance"] > 0]["relevance"].mean(),
            alpha=0.5,
            c="grey",
        )  # the positive score mean
        plt.axvline(
            x=dat[dat["relevance"] < 0]["relevance"].mean(), alpha=0.5, c="grey"
        )  # the negative score mean

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved LRP Barplot to {save_path}")
    # plt.show()
    plt.close(fig)


def plot_aggr_overall_feature_relevances(analysis, features, plt_info, split_path):
    """
    Plots summed relevance of each feature averaged over set of samples.

    Parameters
    ----------
    analysis: the relevance scores
    features: the features
    plt_info: the info df
    split_path: the split path

    Returns
    -------

    """
    for subject in plt_info["sub"].unique():  # ["sub_18"]
        pos_indices = plt_info[
            ((plt_info["sub"] == subject) & (plt_info["y_true"] == 1))
        ].index
        neg_indices = plt_info[
            ((plt_info["sub"] == subject) & (plt_info["y_true"] == 0))
        ].index
        scores_pos = analysis[pos_indices, :, :]
        scores_neg = analysis[neg_indices, :, :]

        # relevance sum for each feature for the pos/neg samples
        relevance_sums_pos = np.sum(scores_pos, axis=1)
        relevance_sums_neg = np.sum(scores_neg, axis=1)
        # then average over samples to get relevance sum of each feature
        relevance_sums_pos = np.mean(relevance_sums_pos, axis=0)
        relevance_sums_neg = np.mean(relevance_sums_neg, axis=0)

        save_path = os.path.join(
            split_path, "innvestigate_plots", "feature_relevance_barplots"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plot_overall_feature_relevance(
            relevance_sums_pos,
            features,
            os.path.join(
                save_path,
                f"{pos_indices[0]}-{pos_indices[-1]}_pos_overall_feature_relevance.pdf",
            ),
            plt_info.iloc[pos_indices],
            f"Aggr. {pos_indices[0]}-{pos_indices[-1]}",
        )
        plot_overall_feature_relevance(
            relevance_sums_neg,
            features,
            os.path.join(
                save_path,
                f"{neg_indices[0]}-{neg_indices[-1]}_neg_overall_feature_relevance.pdf",
            ),
            plt_info.iloc[neg_indices],
            f"Aggr. {neg_indices[0]}-{neg_indices[-1]}",
        )


def plot_lrp_timeseries_aggr(
    y: np.ndarray,
    scores: np.ndarray,
    feature_name: str,
    info: pd.DataFrame,
    gait_events: pd.DataFrame,
    gait_event_scaling_factor_dict: dict,
    # timestamp_col: list,
    # cfg,
    dataset: str,
    save_path=None,
):
    """
    Plots timeseries of many (all) validation samples of a  specirfic subject and condition which gets aggregated by
    e.g. mean value and LRP color coded.
    Parameters
    ----------
    x
    scores
    feature_name
    info
    gait_events
    save_path

    Returns
    -------

    """

    # choose subject (and condition) and plot for each sub their aggregated strides
    subs = info["sub"].unique()
    for subject in subs:
        # get gait events from core_params_py_n.csv for that sub
        dirname = os.path.dirname
        digits_to_round = 3
        # gait_events = {}
        # gait_events["lf_pos"], gait_events["rf_pos"] = get_gait_events(
        #     cfg, dirname, subject, 1, digits_to_round
        # )
        # gait_events["lf_neg"], gait_events["rf_neg"] = get_gait_events(
        #     cfg, dirname, subject, 0, digits_to_round
        # )

        pos_indices = info[((info["sub"] == subject) & (info["y_true"] == 1))].index
        neg_indices = info[((info["sub"] == subject) & (info["y_true"] == 0))].index
        y_pos = y[pos_indices]
        y_neg = y[neg_indices]
        scores_pos = scores[pos_indices]
        scores_neg = scores[neg_indices]

        gait_events = gait_events.reset_index(drop=True)
        gait_events_pos = gait_events.loc[pos_indices]
        gait_events_neg = gait_events.loc[neg_indices]

        # # get the indices of IC and FO for LF/RF
        # # TODO: is this correct? idk. Maybe try using a np.ndarray from the beginning (use padding), to have one
        # #  timestamp for each y_value and then replace the timestamps with 0 or 1 depending if it is a gait event or not
        # events = {}
        # events["LF_IC_neg"] = map_gait_events(
        #     timestamps_neg, gait_events, "lf_neg", "ic", digits_to_round
        # )
        # events["RF_IC_neg"] = map_gait_events(
        #     timestamps_neg, gait_events, "rf_neg", "ic", digits_to_round
        # )
        # events["LF_IC_pos"] = map_gait_events(
        #     timestamps_pos, gait_events, "lf_pos", "ic", digits_to_round
        # )
        # events["RF_IC_pos"] = map_gait_events(
        #     timestamps_pos, gait_events, "rf_pos", "ic", digits_to_round
        # )

        # events["LF_FO_neg"] = map_gait_events(
        #     timestamps_neg, gait_events, "lf_neg", "fo", digits_to_round
        # )
        # events["RF_FO_neg"] = map_gait_events(
        #     timestamps_neg, gait_events, "rf_neg", "fo", digits_to_round
        # )
        # events["LF_FO_pos"] = map_gait_events(
        #     timestamps_pos, gait_events, "lf_pos", "fo", digits_to_round
        # )
        # events["RF_FO_pos"] = map_gait_events(
        #     timestamps_pos, gait_events, "rf_pos", "fo", digits_to_round
        # )

        # # workaround: correct issue #43 (LF IC is not in first point in a stride but in last point of the previous stride)
        # for list in [events["LF_IC_neg"], events["LF_IC_pos"]]:
        #     for series in list:
        #         if series.iloc[0] == 0:
        #             series.iloc[0] = 1
        #         if series.iloc[-1] == 1:
        #             series.iloc[-1] = 0

        # # calculate the average index position in a stride for the events LF_IC, RF_IC, LF_FO, RF_FO
        # avg_event_pos = {}
        # for key in events.keys():
        #     df = pd.DataFrame([series.tolist() for series in events[key]])
        #     df = df.fillna(value=0)
        #     averages_arr = np.array(np.mean(df, axis=0))
        #     positions = np.arange(len(averages_arr))
        #     avg_event_pos[key] = np.round(np.dot(averages_arr, positions))

        # # calculate the avg length (equals the later position of last LF IC line to show end of signal)
        # avg_len_pos, avg_len_neg = (None, None)
        # avg_len_pos = calc_avg_len(events["LF_IC_pos"])
        # avg_len_neg = calc_avg_len(events["LF_IC_neg"])

        # LF_IC_neg_test = pd.DataFrame([series.tolist() for series in events["LF_IC_neg"]])
        # LF_IC_neg_test = LF_IC_neg_test.fillna(value=0)
        # np.mean(LF_IC_neg_test, axis=0)
        # for key in events.keys():
        #     for timeseries in events[key]:
        #         for idx, value in timeseries:
        #             if idx ==

        # ... calculate mean value for all the samples
        y_pos_mean = np.mean(y_pos, axis=0)
        y_neg_mean = np.mean(y_neg, axis=0)

        # ... aggregate lrp value over all samples
        scores_pos_mean = np.mean(scores_pos, axis=0)
        scores_neg_mean = np.mean(scores_neg, axis=0)

        # ... calculate mean gait events
        gait_events_pos_mean = gait_events_pos.loc[pos_indices].mean()
        gait_events_neg_mean = gait_events_neg.loc[neg_indices].mean()

        # only keep positive values from the mean gait events
        gait_events_pos_mean = gait_events_pos_mean[gait_events_pos_mean >= 0]
        gait_events_neg_mean = gait_events_neg_mean[gait_events_neg_mean >= 0]

        # scale gait events for plotting
        gait_events_pos_mean = gait_events_pos_mean * gait_event_scaling_factor_dict[1]
        gait_events_neg_mean = gait_events_neg_mean * gait_event_scaling_factor_dict[0]

        # ... plot line (cf. plot_lrp_timeseries) + sd bars of y

        # calculate SD bars
        sd_bars_pos = get_sd_bars(y_pos, y_pos_mean)
        sd_bars_neg = get_sd_bars(y_neg, y_neg_mean)

        if dataset == "charite":
            paretic_side_text = f"Paretic Side:"
        else:
            paretic_side_text = ""
        title = (
            f"{feature_name} of {subject} Aggregated Strides (y_true=1),"
            + f"{paretic_side_text} {info['paretic_side'].iloc[0]}"
        )  # #{pos_indices[0]}-{pos_indices[-1]};
        plot_lrp_timeseries(
            y_pos_mean,
            scores_pos_mean,
            feature_name,
            info.loc[pos_indices],
            f"{pos_indices[0]}-{pos_indices[-1]}",
            gait_events_pos_mean,
            dataset,
            save_path,
            title,
            # avg_event_pos,
            sd_bars=sd_bars_pos,
            # avg_length=avg_len_pos,
        )
        title = (
            f"{feature_name} of {subject} Aggregated Strides (y_true=0), "
            + f"{paretic_side_text} {info['paretic_side'].iloc[0]}"
        )  # #{neg_indices[0]}-{neg_indices[-1]};
        plot_lrp_timeseries(
            y_neg_mean,
            scores_neg_mean,
            feature_name,
            info.loc[neg_indices],
            f"{neg_indices[0]}-{neg_indices[-1]}",
            gait_events_neg_mean,
            dataset,
            save_path,
            title,
            # avg_event_pos,
            sd_bars=sd_bars_neg,
            # avg_length=avg_len_neg,
        )

    # TODO: Problem: for each fixed sub and fatigue status, we only have about 10 samples in the validation data doing
    #  intra CV:
    #  a) make less splits
    #  b) Maybe we should switch the evaluation scenario to having a fixed big test set (intra: data from all subs)
    pass


def get_sd_bars(y, y_mean):
    """Calculates the standard deviation bars for the provided y vector and returns it as a dataframe."""
    sd_bars_df = pd.DataFrame()  # columns=["y_min", "y_max"]
    y_sd = np.std(y, axis=0)
    sd_bars_df["x"] = np.array(range(0, y.shape[1]))
    sd_bars_df["y_min"] = y_mean - y_sd
    sd_bars_df["y_max"] = y_mean + y_sd

    return sd_bars_df


def plot_lrp_timeseries(
    x,
    scores,
    feature_name,
    info,
    sample_idx,
    gait_events_mean,
    dataset,
    save_path=None,
    title=None,
    # event_positions=None,
    sd_bars=None,
    # avg_length=None,
):
    """
    Plots timeseries and highlights spots according to relevance.

    Parameters
    ----------
    x: the time series of shape (timesteps, features)
    scores: relevance scores of shape (timesteps, features)
    feature_name: the name of the feature
    info: df containing sample information such as "sub", "y_true", "y_pred" that are useful for interpretation
    sample_idx: the index of the sample in the validation data
    gait_events_mean: the mean gait events
    save_path: the path to save the plot to
    title: the figure title
    sd_bars: the standard deviation bars list of (index-start-end) triples
    avg_length: the average length needed for aggr lrp timeseries plots to mark the position of the ending LF IC event

    Returns
    -------

    """
    font_size = 20
    params = {
        "legend.fontsize": font_size,  # 'x-large'
        # 'figure.figsize': (15, 5),
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    }
    pylab.rcParams.update(params)

    dat = pd.DataFrame(data={"x": x, "scores": scores})

    # create line segments in matplotlib
    # see https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    y = x
    x = np.array(dat.index)

    # cut plot at longest stride
    last_y = y[-1]
    last_x = x[-1]
    if (
        y[-1] != y[-2]
    ):  # max stride length is actually reached here (i.e., no padding with avg values)
        if sd_bars is not None:
            sd_bars = sd_bars[:-2]
        pass
    else:
        # find longest stride index
        for x_i, y_i in zip(reversed(x), reversed(y)):
            if y_i == last_y:
                last_x = x_i
                continue
            else:
                # actual values start here => cut
                y = y[:last_x]
                x = x[:last_x]
                scores = scores[:last_x]
                if sd_bars is not None:
                    sd_bars = sd_bars[:last_x]
                break

    # create line segments (two points/x-y-pairs for each line)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    plt.figure(figsize=(6, 10))
    fig, axs = plt.subplots()
    fig.set_size_inches(16, 8)  # 20.8, 8 for AIME paper sub01 AccX_LF

    # norm to map for datapoint -> color mapping
    norm = plt.Normalize(
        scores.min(), scores.max()
    )  # the legend scale on the plot displays non-normalized values
    lc = LineCollection(segments, cmap="turbo", norm=norm)  # viridis
    # Set the values used for colormapping
    lc.set_array(scores)
    lc.set_linewidth(4)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs, label="LRP Score")

    # y_min = round_down_to_point_five(y.min())
    # y_max = round_up_to_point_five(y.max())
    # axs.set_ylim(y_min, y_max)
    # axs.set_ylim(int(math.floor(y.min())), int(math.ceil(y.max())))
    plt.xlabel("Index")
    if "acc" in feature_name:
        ylabel = "Acc. [g]"
    plt.ylabel(feature_name)

    if sd_bars is not None:
        y_min = math.floor(sd_bars["y_min"].min())
        y_max = math.ceil(sd_bars["y_max"].max())
        axs.set_ylim(y_min, y_max)
        sd_bars.apply(
            lambda x: plt.vlines(
                x=x["x"],
                ymin=x["y_min"],
                ymax=x["y_max"],
                colors=plt.cm.turbo(norm(scores[int(x["x"])])),
                lw=1,
                alpha=0.8,
            ),
            axis=1,
        )  # colors='grey'

    if not title:
        sub = info["sub"]
        y_true = info["y_true"]
        y_pred = info["y_pred"]
        title = f"{feature_name} with colored LRP relevances ({sub}, y_true={y_true}, y_pred = {y_pred}, sample {sample_idx})"
    # fig.suptitle(title, fontsize=18)  #  y=plt.gca().get_ylim()[1] + 0.1

    # adjust axes manually
    # axs.set_xlim(0, 170)  # 175
    # axs.set_ylim(-1.0, 1.1)  # (-0.7, 0.5)

    # rename gait events for plotting
    gait_events_mean = gait_events_mean.rename(
        index={
            "ic1": "Initial Contact 1",
            "fo_time": "Foot Off",
            "ic2": "Initial Contact 2",
        }
    )
    for gait_event in ["Initial Contact 1", "Foot Off", "Initial Contact 2"]:
        # plot the vertical line for gait event
        if gait_event in gait_events_mean.index:
            plt.axvline(
                int(gait_events_mean[gait_event]),
                color="grey",
                linestyle="--",
                # label=gait_event,
            )
            # add text for label above the lines
            plt.text(
                int(gait_events_mean[gait_event]),
                plt.gca().get_ylim()[1]
                + 0.02 * (plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]),
                gait_event,
                alpha=0.5,
                fontsize=font_size,
                ha="center",
                # rotation=45,
            )

    # plot vlines for gait events
    # if event_positions:
    #     long_event_name = {
    #         "LF_IC": "LF Init. Contact",
    #         "RF_IC": "RF Init. Contact",
    #         "LF_FO": "LF Foot Off",
    #         "RF_FO": "RF Foot Off",
    #     }

    #     if (
    #         info["y_true"].iloc[0] == 1
    #     ):  # 1 if for positive run (e.g., fatigue), 0 for negative (e.g., control)
    #         run = "pos"
    #     else:
    #         run = "neg"

    #     for key in event_positions.keys():
    #         if not run in key:
    #             continue
    #         # if ("LF" in key and "RF" in title) or ("RF" in key and "LF" in title):  # for LF, only plot LF (and RF analogue)
    #         #     continue
    #         if "LF" in key:
    #             color = "grey"
    #         elif "RF" in key:
    #             color = "purple"
    #         plt.axvline(x=event_positions[key], c=color, alpha=0.5, ls="--")
    #         # plt.text(event_positions[key], plt.gca().get_ylim()[1] + 0.03, key[:-3].replace("_", " "), alpha=0.5, fontsize="x-large", rotation=45)
    #         plt.text(
    #             event_positions[key],
    #             plt.gca().get_ylim()[1] + 0.03,
    #             long_event_name[key[:-4]],
    #             alpha=0.5,
    #             fontsize=font_size,
    #             rotation=45,
    #         )

    #     # # line pos at avg last position of the strides
    #     if avg_length:
    #         plt.axvline(x=int(avg_length), c="grey", alpha=0.5, ls="--")
    #         plt.text(
    #             int(avg_length),
    #             plt.gca().get_ylim()[1] + 0.03,
    #             "LF Init. Contact",
    #             alpha=0.5,
    #             fontsize=font_size,
    #             rotation=45,
    #         )
    plt.title(title, y=1.09)
    plt.tight_layout()
    if save_path:
        save_path = os.path.join(save_path, "innvestigate_plots")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path_to_file = os.path.join(
            save_path, f"{sample_idx}_{feature_name}_relevance.pdf"
        )
        plt.savefig(path_to_file, bbox_inches="tight")
        print(f"saved LRP plot to {path_to_file} for sample {sample_idx}.")
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # import paths
    dataset = "charite"  #"duo_gait"  # "charite"
    with open("path.json", "r") as f:
        paths = json.load(f)
    original_data_path = paths[f"data_{dataset}_original"]
    data_base_path = paths[f"data_{dataset}"]

    exp_name = "2024-10-20_15-20-21"  # "2024-04-12_17-09-47_LF_RF_resample"

    if dataset == "duo_gait":
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
    elif dataset == "charite":
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

    for sub in sub_list:

        # load the config
        config = json.load(
            open(
                os.path.join(
                    data_base_path, "results", exp_name, sub, "config_sub.json"
                ),
                "r",
            )
        )

        # load the model
        model_dir = os.path.join(
            data_base_path,
            "results",
            exp_name,
            sub,
        )
        model = tf.keras.models.load_model(os.path.join(model_dir, "best_model.h5"))

        # Stripping the softmax activation from the model
        model_wo_sm = innvestigate.model_wo_softmax(model)

        # load the scaler
        scaler_path = os.path.join(model_dir, "train_val_scaler.pkl")
        scaler_dict = pickle.load(open(scaler_path, "rb"))

        # load the test data
        test_data_dir = os.path.join(data_base_path, "test_data", sub)
        test_data = pd.read_csv(os.path.join(test_data_dir, "test_data.csv"))

        # prepare the data
        # cut the test data into windows
        feature_builder = FeatureBuilder(
            original_data_path=original_data_path,
            data_base_path=data_base_path,
            data_df=test_data,
            config=config,
            scaler_dict=scaler_dict,
        )

        feature_builder.normalize_2d_data(plot_title=f"{sub} test data")
        max_window_size = model.input_shape[
            1
        ]  # get the window size from the input shape of the model
        X_test, y_test, gait_events_dict_test = feature_builder.make_windows(
            max_window_size=max_window_size
        )

        features = feature_builder.get_feature_names()

        #### Prepare gait events for plotting
        if "resample" in exp_name:
            # get the gait event scaling factor for plotting
            gait_event_scaling_factor_dict = (
                feature_builder.get_gait_event_scaling_factors()
            )

            # # divide the scaling factor by sampling rate (prepare for plotting)
            # gait_event_scaling_factor_dict = {
            #     key: value / sampling_rate for key, value in gait_event_scaling_factor_dict.items()
            # }

        else:
            if dataset == "duo_gait":
                sampling_rate = 128
            elif dataset == "charite":
                sampling_rate = 120
            gait_event_scaling_factor_dict = {}
            for key in np.unique(y_test):
                gait_event_scaling_factor_dict[key] = sampling_rate

        # Analyze model
        model.predict(X_test)

        analyzer = innvestigate.create_analyzer(
            "lrp.epsilon",
            model_wo_sm,
            neuron_selection_mode="max_activation",
            **{"epsilon": 1},
        )
        analysis = analyzer.analyze(X_test)
        # for y_pred, undo the one-hot encoding

        if dataset == "charite":
            paretic_side = config["paretic_side"]
        elif dataset == "duo_gait":
            paretic_side = ""
        plt_info = pd.DataFrame(
            data={
                "sub": sub,
                "paretic_side": paretic_side,
                "y_true": y_test,
                "y_pred": np.argmax(model.predict(X_test), axis=-1),
            }
        )

        # bar plots of all single samples
        # plot_overall_feature_relevance_for_sub(
        #     analysis,
        #     features,
        #     plt_info,
        #     os.path.join(data_base_path, "results", exp_name, sub),
        #     sub,
        # )

        # bar plots of aggregated feature relevances
        plot_aggr_overall_feature_relevances(
            analysis,
            features,
            plt_info,
            os.path.join(data_base_path, "results", exp_name, sub),
        )

        for feature in range(len(features)):  # for each feature
            feature_name = features[feature]  # e.g. 'GyrX_LF'
            # For dynamic features, use the corresponding scaler of the sensor (e.g., 'LF').
            # For static features, any scaler has the same mean and var values, so any can be used
            # scaler, sensor = self._get_scaler(feature_name)

            suffix = ""
            # if sensor:  # feature is sensor dependent (i.e., dynamic)
            #     # remove the sensor suffix (e.g., '_LF')
            #     suffix = feature_name[-3:]
            #     feature_name = feature_name[:-3]
            # # get index of the sensor feature to extract the correct mean and variance values from the scaler
            idx = features.index(
                feature_name
            )  # the idx of the feature in the sensor_features list

            # denormalize the data vector using the class-specific scaler
            x = X_test[:, :, feature]
            x_class_ls = []
            for class_label in plt_info["y_true"].unique():
                # select data from the class
                class_idx = plt_info[plt_info["y_true"] == class_label].index
                x_class = x[class_idx]

                # denormalize the data
                dummy_2d = np.ones(
                    (len(class_idx) * x_class.shape[1], len(features))
                )  # to fit the shape requirements of scaler
                dummy_2d[:, idx] = x_class.reshape(-1)
                dummy_2d_denormalized = scaler_dict[class_label].inverse_transform(
                    dummy_2d
                )
                x_class = dummy_2d_denormalized[:, idx].reshape(
                    x_class.shape[0], x_class.shape[1]
                )
                x_class_ls.append(x_class)

            # concatenate data from all classes
            x = np.concatenate(x_class_ls, axis=0)

            scores = analysis[:, :, feature]

            # df_concat["LF_IC"] = timestamp_col.map(lambda x: 1 if (np.round(x, digits_to_round) in np.array(gait_events_lf["ic_times"]) or np.round(x, digits_to_round) == np.round(gait_events_lf["timestamps"].iloc[0], digits_to_round)) else 0)
            # df_concat["LF_FO"] = timestamp_col.map(lambda x: 1 if np.round(x, digits_to_round) in np.array(gait_events_lf["fo_times"]) else 0)
            # df_concat["RF_IC"] = timestamp_col.map(lambda x: 1 if (np.round(x, digits_to_round) in np.array(gait_events_rf["ic_times"]) or np.round(x, digits_to_round) == np.round(gait_events_rf["timestamps"].iloc[0], digits_to_round)) else 0)
            # df_concat["RF_FO"] = timestamp_col.map(lambda x: 1 if np.round(x, digits_to_round) in np.array(gait_events_rf["fo_times"]) else 0)

            # select gait events for the time series plot
            if np.logical_and(
                np.logical_or("LF" in feature_name, "RF" in feature_name),
                config["segmentation_sensor"] not in feature_name,
            ):  # if it is the other foot, use the other foot's gait events
                side = "other"
            else:  # use segmentation foot's gait events for all other cases
                side = "segmentation"

            gait_events_test = gait_events_dict_test[side]

            plot_lrp_timeseries_aggr(
                x,
                scores,
                feature_name=feature_name + suffix,
                info=plt_info,
                gait_events=gait_events_test,
                gait_event_scaling_factor_dict=gait_event_scaling_factor_dict,
                # timestamp_col=timestamp_col,
                # cfg=cfg,
                dataset=dataset,
                save_path=os.path.join(data_base_path, "results", exp_name, sub),
            )
