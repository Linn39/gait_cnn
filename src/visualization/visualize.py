import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.patches import Patch
# from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

    Args:
        cv (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        group (_type_): _description_
        ax (_type_): _description_
        n_splits (_type_): _description_
        lw (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    rng = np.random.RandomState(1338)
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    
    # create a LabelEncoder object to encode the groups column
    le = LabelEncoder()

    # fit and transform the "sub" column
    group = le.fit_transform(group)

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
        # xlim=[0, len(X)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax