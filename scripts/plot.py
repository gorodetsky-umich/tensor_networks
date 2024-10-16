"""Plotting scripts."""

import argparse
import json
import itertools

import matplotlib.pyplot as plt

all_figures = [
    # time
    "time_box",
    # error
    "error_by_time",
    # compression
    "cost_by_time",
    "best_cost_by_time",
    # other
    "ops_by_time",
]

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, help="input data")
parser.add_argument("--output", type=str, help="output file prefix")
parser.add_argument("--figures", choices=all_figures, nargs='+')


def plot_time_box(tag, filename):
    """Plot the time distribution as a box plot."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = []
        for l in lines:
            fields = l.split(', ')
            if fields[2] != "1.0" and float(fields[1]) < 500:
                data.append(float(fields[1]))

        _, ax = plt.subplots()
        ax.boxplot(data, meanline=True, tick_labels=["Time"])
        ax.set_ylabel('Time (s)')
        plt.savefig(f"{tag}_time_box.png")

def plot_x_by_time(tag, filename, figure_type, column_tag):
    """Make the line plot. 
    
    x axis: time
    y axis: data from the column tag
    """
    with open(filename, "r", encoding="utf-8") as f:
        all_stats = json.load(f)
        stats = all_stats[-1]
        time, data = list(zip(*stats[column_tag]))
        _, ax = plt.subplots(1, 1)
        ax.plot(time, data, marker=".")
        ax.hlines([max(data), min(data)], xmin=time[0], xmax=time[-1], colors='red', linestyles='dashed')
        plt.yticks([x for x in list(plt.yticks()[0]) if x >= 0] + [max(data), min(data)])

        # zoom into the details
        # ax_lower.plot(time[100:], data[100:], marker=".")

        plt.savefig(f"{tag}_{figure_type}.png")
        plt.close()

def plot_error_box(tag, filename):
    """Make a box plot.
    
    x axis: kinds of boxes
    y axis: errors in percentage
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        re_data, compression_data = [], []
        for l in lines:
            fields = l.split(', ')
            if fields[2] != "1.0" and float(fields[1]) < 500:
                re_data.append(float(fields[2]) * 100)
                compression_data.append(float(fields[5]) * 100)

        _, ax = plt.subplots()
        ax.boxplot([re_data, compression_data], meanline=True, tick_labels=["Construction Error", "Compression Rate"])
        ax.set_ylabel('Construction Error and Compression Rate (%)')
        plt.savefig(f"{tag}_error.png")


if __name__ == "__main__":
    args = parser.parse_args()
    plots = []
    if "error_by_time" in args.figures:
        plot_x_by_time(args.output, args.file, "error_by_time", "errors")

    if "cost_by_time" in args.figures:
        plot_x_by_time(args.output, args.file, "cost_by_time", "costs")

    if "ops_by_time" in args.figures:
        plot_x_by_time(args.output, args.file, "ops_by_time", "ops")

    if "best_cost_by_time" in args.figures:
        plot_x_by_time(args.output, args.file, "best_cost_by_time", "best_cost")
