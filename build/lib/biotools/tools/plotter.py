import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class DimensionalityReductionPlotter:
    """
    A class to easily apply multiple t-SNE jobs (with default perplexities) or a single PCA,
    create scatter plots of the results, and generate KDE plots.
    """

    # Default TSNE jobs as (perplexity, n_iter)
    DEFAULT_TSNE_JOBS = [
        (1, 1000),
        (2, 1000),
        (5, 1000),
        (10, 1000),
        (50, 1000),
        (100, 1000),
    ]

    def __init__(self, output_dir: str = "outputs"):
        """
        :param output_dir: Directory where plots (and any pickled data) will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_tsne(
        self,
        data: torch.Tensor,
        hue_column: Optional[pd.Series] = None,
        shape_column: Optional[pd.Series] = None,
        line_group_column: Optional[pd.Series] = None,
        title_prefix: str = "TSNE",
        random_state: int = 42,
        save_data: bool = True,
    ) -> None:
        """
        Runs multiple t-SNE configurations (using default perplexities) on the given data,
        then plots each result.

        :param data: PyTorch tensor of shape (N, D).
        :param hue_column: Optional Series for coloring points.
        :param shape_column: Optional Series for shaping points.
        :param line_group_column: Optional Series for connecting points in the same group.
        :param title_prefix: String to use as the beginning of plot titles.
        :param random_state: Random seed for t-SNE.
        :param save_data: If True, pickle the final 2D results for each run.
        """
        # Convert torch.Tensor to NumPy
        np_data = data.detach().cpu().numpy()

        for (perplexity, n_iter) in self.DEFAULT_TSNE_JOBS:
            # 1) Run TSNE
            tsne_result = self._run_tsne(
                np_data,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=random_state,
                n_components=2,
            )

            # 2) Create a scatter plot
            file_name = f"tsne_perp{perplexity}_iter{n_iter}.png"
            self._plot_scatter(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                hue=hue_column,
                shape=shape_column,
                line_group=line_group_column,
                title=f"{title_prefix}: Perp={perplexity}, Iter={n_iter}",
                file_name=file_name,
            )

            # 3) Optionally save data to a pickle file
            if save_data:
                self._save_to_pickle(
                    tsne_result,
                    os.path.join(self.output_dir, f"tsne_perp{perplexity}_iter{n_iter}.pkl"),
                )

    def plot_pca(
        self,
        data: torch.Tensor,
        hue_column: Optional[pd.Series] = None,
        shape_column: Optional[pd.Series] = None,
        line_group_column: Optional[pd.Series] = None,
        title: str = "PCA",
        n_components: int = 2,
        save_data: bool = True,
    ) -> None:
        """
        Runs PCA (2D by default) on the given data, then plots the result.

        :param data: PyTorch tensor of shape (N, D).
        :param hue_column: Optional Series for coloring points.
        :param shape_column: Optional Series for shaping points.
        :param line_group_column: Optional Series for connecting points in the same group.
        :param title: Title for the PCA plot.
        :param n_components: Number of PCA components (default=2).
        :param save_data: If True, pickle the final 2D results.
        """
        np_data = data.detach().cpu().numpy()

        # 1) Run PCA
        pca_result = self._run_pca(np_data, n_components=n_components)

        # 2) Create a scatter plot
        file_name = f"pca_{n_components}d.png"
        self._plot_scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            hue=hue_column,
            shape=shape_column,
            line_group=line_group_column,
            title=title,
            file_name=file_name,
        )

        # 3) Optionally save data
        if save_data:
            self._save_to_pickle(
                pca_result,
                os.path.join(self.output_dir, f"pca_{n_components}d.pkl"),
            )

    def plot_kde(
        self,
        distributions: List[torch.Tensor],
        dist_names: List[str],
        colors: Optional[List[str]],
        plot_title: str,
        file_name: str,
        x_label: str,
        y_label: str = "Density",
        mean_line: bool = False,
        median_line: bool = False,
        fill_graphs: bool = True,
        normalize: bool = False,
        save_data: bool = True,
    ) -> None:
        """
        Plot one or more distributions as KDE plots.

        :param distributions: List of PyTorch Tensors.
        :param dist_names: List of names for each distribution.
        :param colors: List of color codes or None; must match len(distributions).
        :param plot_title: Title of the plot.
        :param file_name: Output file name.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis (default: "Density").
        :param mean_line: If True, draw a vertical line at the mean of each distribution.
        :param median_line: If True, draw a vertical line at the median of each distribution.
        :param fill_graphs: If True, fill the area under the KDE curve.
        :param normalize: If True, min-max normalize each distribution before plotting.
        :param save_data: If True, pickle the distribution data.
        """
        # Basic checks
        if colors is None:
            colors = sns.color_palette("husl", len(distributions))
        if len(distributions) != len(colors):
            raise ValueError("Distributions and colors must have the same length.")

        plt.figure(figsize=(12, 8), constrained_layout=True)

        for dist, name, color in zip(distributions, dist_names, colors):
            dist = dist.cpu()

            if normalize:
                dist = (dist - dist.min()) / (dist.max() - dist.min())

            sns.kdeplot(
                dist.numpy(),
                color=color,
                fill=fill_graphs,
                label=name,
            )

            if mean_line:
                m = dist.mean().item()
                plt.axvline(
                    x=m,
                    color=color,
                    linestyle="--",
                    label=f"Mean {name} ({m:.2f})",
                )

            if median_line:
                med = dist.median().item()
                plt.axvline(
                    x=med,
                    color=color,
                    linestyle="-",
                    label=f"Median {name} ({med:.2f})",
                )

        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

        out_path = os.path.join(self.output_dir, file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()

        if save_data:
            pickle_path = os.path.splitext(out_path)[0] + ".pkl"
            self._save_to_pickle(distributions, pickle_path)

    #######################
    # Internal Helper Methods
    #######################

    def _run_tsne(
        self,
        data: np.ndarray,
        perplexity: int,
        n_iter: int,
        random_state: int,
        n_components: int = 2,
    ) -> np.ndarray:
        """
        Apply t-SNE to reduce the dimensionality of data.
        """
        tsne = TSNE(
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            n_components=n_components,
            verbose=0,
        )
        return tsne.fit_transform(data)

    def _run_pca(self, data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Apply PCA to reduce the dimensionality of data.
        """
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data)

    def _plot_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        hue: Optional[pd.Series],
        shape: Optional[pd.Series],
        line_group: Optional[pd.Series],
        title: str,
        file_name: str,
        alpha: float = 0.8,
        marker_size: float = 40,
        palette: Optional[List[str]] = None,
    ):
        """
        Internal method that actually creates and saves a scatter plot.
        """

        df = pd.DataFrame({"x": x, "y": y})
        if hue is not None:
            df["hue"] = hue.values if isinstance(hue, pd.Series) else hue
        if shape is not None:
            df["shape"] = shape.values if isinstance(shape, pd.Series) else shape
        if line_group is not None:
            df["line_group"] = line_group.values if isinstance(line_group, pd.Series) else line_group

        # Sort by hue to ensure consistent color legend ordering
        if "hue" in df.columns:
            df.sort_values(by="hue", inplace=True)

        # If no custom palette is provided, use colorcet's glasbey for distinct categories
        if palette is None and "hue" in df.columns:
            unique_hues = df["hue"].nunique()
            palette = sns.color_palette(cc.glasbey, n_colors=unique_hues)

        plt.figure(figsize=(10, 6))

        if "shape" in df.columns:
            # Build a size mapping dict so the first shape is 40, next is 60, etc.
            unique_shapes = df["shape"].unique()
            size_dict = {}
            for i, cat in enumerate(unique_shapes):
                size_dict[cat] = 40 + (i * 20)  # 40, 60, 80, ...

            # Single call to scatterplot, letting Seaborn handle shape differently.
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="hue" if "hue" in df.columns else None,
                style="shape",
                size="shape",
                sizes=size_dict,
                alpha=alpha,
                palette=palette,
                legend="full",
                markers=True,
            )
        else:
            # Single scatter call if no shape
            sns.scatterplot(
                data=df,
                x="x",
                y="y",
                hue="hue" if "hue" in df.columns else None,
                alpha=alpha,
                s=marker_size,
                palette=palette,
            )

        # Draw lines for points that share a line_group
        if "line_group" in df.columns:
            for grp, sub_df in df.groupby("line_group"):
                if len(sub_df) > 1:
                    plt.plot(
                        sub_df["x"],
                        sub_df["y"],
                        linestyle=(0, (5, 5)),
                        linewidth=1,
                        color="black",
                    )

        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.subplots_adjust(right=0.75)

        # If we have hue or shape, show legend
        if "hue" in df.columns or "shape" in df.columns:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)

        out_path = os.path.join(self.output_dir, file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    def _save_to_pickle(self, data, path: str):
        """
        Save data to a pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(data, f)
