"""Headless, reproducible EDA figures from measured data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def plot_inventory(inventory, destination):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for partition, color in [("train", "#167d9a"), ("student_test", "#e89c35")]:
        group = inventory[inventory.partition == partition]
        axes[0, 0].hist(group.gps_observations, bins=15, alpha=.65, label=partition, color=color)
        axes[0, 1].scatter(group.nodes, group.arcs, label=partition, color=color, alpha=.75)
        axes[1, 0].scatter(group.gps_observations, group.max_gap_s, color=color, alpha=.75)
        axes[1, 1].hist(group.weak_components, bins=12, alpha=.65, label=partition, color=color)
    axes[0, 0].set(xlabel="GPS observations per trip", ylabel="Trips", title="Trajectory sizes")
    axes[0, 1].set(xlabel="Nodes per trip", ylabel="Directed arcs per trip", title="Network sizes")
    axes[1, 0].set(xlabel="GPS observations per trip", ylabel="Largest time gap (s)", title="Sampling anomalies")
    axes[1, 1].set(xlabel="Weakly connected components", ylabel="Trips", title="Network connectivity")
    axes[0, 0].legend()
    fig.suptitle("PathSnap | Dataset inventory", fontsize=17)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_raw_trip(trip, track, network, destination):
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    segments = network.nodes_xy[network.endpoints]
    lo = track[["x_m", "y_m"]].min().to_numpy() - 200
    hi = track[["x_m", "y_m"]].max().to_numpy() + 200
    visible = ((segments.max(axis=1) >= lo) & (segments.min(axis=1) <= hi)).all(axis=1)
    ax.add_collection(LineCollection(segments[visible], colors="#b8c3cb", linewidths=.6, alpha=.65))
    if trip.route is not None:
        ax.add_collection(LineCollection(segments[trip.route.arc_id.to_numpy()], colors="#28935c", linewidths=2,
                                         label="Supplied reference route"))
    ax.scatter(track.x_m, track.y_m, s=5, c="#136bc0", label="Raw GPS", zorder=3)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel=f"Easting (m), EPSG:{network.projection.epsg}",
           ylabel="Northing (m)", title=f"{trip.partition} / {trip.trip_id} | GPS and road network")
    ax.set_aspect("equal")
    ax.legend()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_feature_summary(features, labels, destination):
    from src.features.reference_labeling import build_training_table
    merged = build_training_table(features, labels)
    eligible = merged.loc[merged.training_eligible]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for target, color, title in [(0, "#d29239", "Non-match"), (1, "#157e99", "Reference proxy match")]:
        group = eligible.loc[eligible.target == target]
        axes[0].hist(group.distance_m, bins=30, density=True, alpha=.6, color=color, label=title)
        axes[1].hist(group.loc[group.heading_valid == 1, "heading_difference_deg"], bins=30,
                     density=True, alpha=.6, color=color)
    axes[0].set(xlabel="Point-to-segment distance (m)", ylabel="Density", title="Distance by proxy class")
    axes[1].set(xlabel="Heading difference (degrees)", ylabel="Density", title="Moving observations only")
    axes[2].hist(features.groupby("observation_id").size(), bins=np.arange(.5, 10.5), color="#157e99")
    axes[2].set(xlabel="Candidates per observation", ylabel="Observations", title="Candidate-set sizes")
    axes[0].legend(fontsize=8)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)
