import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="input data")
parser.add_argument("--tag")


def plot_time(tag, filename):
  with open(filename, "r") as f:
    lines = f.readlines()
    data = []
    for l in lines:
      fields = l.split(', ')
      if fields[2] != "1.0" and float(fields[1]) < 500:
        data.append(float(fields[1]))

    fig, ax = plt.subplots()
    ax.boxplot(data, meanline=True, tick_labels=["Time"])
    ax.set_ylabel('Time (s)')
    plt.savefig(f"{tag}_time.png")

def plot_error(tag, filename):
  with open(filename, "r") as f:
    lines = f.readlines()
    re_data, compression_data = [], []
    for l in lines:
      fields = l.split(', ')
      if fields[2] != "1.0" and float(fields[1]) < 500:
        re_data.append(float(fields[2]) * 100)
        compression_data.append(float(fields[5]) * 100)

    fig, ax = plt.subplots()
    ax.boxplot([re_data, compression_data], meanline=True, tick_labels=["Construction Error", "Compression Rate"])
    ax.set_ylabel('Construction Error and Compression Rate (%)')
    plt.savefig(f"{tag}_error.png")


if __name__ == "__main__":
  args = parser.parse_args()
  plot_time(args.tag, args.file)
  plot_error(args.tag, args.file)