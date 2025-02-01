import os
from typing import List
import pickle
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

DEFAULT_TSNE_JOBS = [
    (1, 1000),
    (2, 1000),
    (5, 1000),
    (10, 1000),
    (50, 1000),
    (100, 1000),
    (1, 5000),
    (2, 5000),
    (5, 5000),
    (10, 5000),
    (50, 5000),
    (100, 5000),
]


def apply_tsne(
    data: np.ndarray, perplexity: int, n_iter: int, random_state: int, n_components=2
):
    tsne = TSNE(
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        n_components=n_components,
        verbose=0,
    )
    return tsne.fit_transform(data)


def apply_pca(data: np.ndarray, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)


def create_scatter_plot(
    scatter_data: pd.DataFrame,
    title: str,
    output_file_path: str,
    x_name="x",
    y_name="y",
    hue_name=None,
    shape_name=None,
    line_group_name=None,
    legend_title=None,
    palette=None,
    # markers=None,
    alpha=1,
    marker_size=40,
):
    if hue_name is not None:
        scatter_data = scatter_data.sort_values(by=hue_name)

    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    if palette is None and hue_name is not None:
        palette = sns.color_palette(
            cc.glasbey, n_colors=scatter_data[hue_name].nunique()
        )

    sns.scatterplot(
        x=x_name,
        y=y_name,
        hue=hue_name,
        style=shape_name,
        palette=palette,
        data=scatter_data,
        legend="full",
        alpha=alpha,
        # markers=markers,
        s=marker_size,
    )

    if line_group_name is not None:
        for group, group_data in scatter_data.groupby(line_group_name):
            if len(group_data) > 1 and group is not None:
                plt.plot(
                    group_data[x_name],
                    group_data[y_name],
                    linestyle=(0, (5, 5)),
                    linewidth=1,
                    color="black",
                )

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.subplots_adjust(right=0.75)

    if hue_name is not None or shape_name is not None:
        plt.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), ncol=2, title=legend_title
        )

    plt.savefig(output_file_path)
    plt.close()


def plot_reduced_dim(
    run_name: str,
    embeddings: torch.Tensor,
    output_folder: str,
    seed: int,
    color_column=None,
    color_column_name=None,
    shape_column=None,
    shape_column_name=None,
    line_group_column=None,
    line_column_name=None,
    legend_title=None,
    jobs=DEFAULT_TSNE_JOBS,
    colors=None,
    # markers=None,
    alpha=1,
    marker_size=40,
):
    os.makedirs(os.path.join(output_folder, run_name), exist_ok=True)

    for perplexity, n_iter in tqdm(jobs, desc="Going through t-SNE jobs"):
        print("Applying t-SNE...")
        try:
            tsne_results = apply_tsne(embeddings.numpy(), perplexity, n_iter, seed)
        except Exception as e:
            print("Failed to apply t-SNE, reason: ", e)
            continue

        print("Finished applying t-SNE!")

        scatter_data = pd.DataFrame(
            {
                "x": tsne_results[:, 0],
                "y": tsne_results[:, 1],
            }
        )

        if color_column is not None:
            scatter_data[color_column_name] = color_column

        if shape_column is not None:
            scatter_data[shape_column_name] = shape_column

        if line_group_column is not None:
            scatter_data[line_column_name] = line_group_column

        create_scatter_plot(
            scatter_data,
            run_name,
            os.path.join(
                output_folder, run_name, f"tsne-{n_iter}iter_{perplexity}perp.png"
            ),
            hue_name=color_column_name,
            shape_name=shape_column_name,
            legend_title=legend_title,
            palette=colors,
            # markers=markers,
            alpha=alpha,
            marker_size=marker_size,
        )


def create_pca_plots_with_shape(
    run_name: str,
    embeddings: torch.Tensor,
    output_folder: str,
    color_column=None,
    color_column_name=None,
    shape_column=None,
    shape_column_name=None,
    line_group_column=None,
    line_column_name=None,
    legend_title=None,
    colors=None,
    # markers=None,
):
    print("Applying PCA...")
    pca_results = apply_pca(embeddings.numpy())
    print("Finished applying PCA!")

    scatter_data = pd.DataFrame(
        {
            "x": pca_results[:, 0],
            "y": pca_results[:, 1],
        }
    )

    if color_column is not None:
        scatter_data[color_column_name] = color_column

    if shape_column is not None:
        scatter_data[shape_column_name] = shape_column

    if line_group_column is not None:
        scatter_data[line_column_name] = line_group_column

    create_scatter_plot(
        scatter_data,
        run_name,
        os.path.join(output_folder, run_name, f"pca.png"),
        hue_name=color_column_name,
        shape_name=shape_column_name,
        legend_title=legend_title,
        palette=colors,
        # markers=markers,
    )


def create_kde_plot(
    distributions: List[torch.Tensor],
    dist_names: List[str],
    colors: List[str],
    plot_title: str,
    output_file: str,
    x_label: str,
    y_label="Density",
    mean_line=False,
    median_line=False,
    fill_graphs=True,
    should_normalize=False,
    save_data=True,
):
    if len(distributions) != len(colors):
        raise Exception(
            f"Received different amounts of distributions ({len(distributions)}) and colors ({len(colors)})"
        )

    plt.figure(figsize=(16, 9), constrained_layout=True)

    for distribution, dist_name, color in zip(distributions, dist_names, colors):

        if should_normalize:
            distribution = (distribution - distribution.min()) / (
                distribution.max() - distribution.min()
            )

        distribution = distribution.cpu()
        sns.kdeplot(
            distribution.numpy(),
            color=color,
            fill=fill_graphs,
            label=dist_name,
        )

        if mean_line:
            mean = torch.mean(distribution).item()

            plt.axvline(
                mean,
                color=color,
                linestyle="--",
                label=f"Mean {dist_name} ({mean:.2f})",
            )

        if median_line:
            median = torch.median(distribution).item()

            plt.axvline(
                median,
                color=color,
                linestyle="-",
                label=f"50% {dist_name} ({median:.2f})",
            )

    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    plt.savefig(output_file)

    if save_data:
        save_to_pickle(distributions, os.path.splitext(output_file)[0] + ".pkl")

    plt.close()


def save_to_pickle(data, path: str):
    with open(path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
