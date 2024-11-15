import glob
import json
import pickle

def collect_tt(filename: str):
    with open(filename, "r") as f:
        line = f.read().strip()
        fields = line.split(' ')
        print(f"{fields[2]}\t{fields[3]}\t{float(fields[5])}")

def collect_ht(filename: str):
    with open(filename, "r") as f:
        line = f.read().split('\n')[1]
        fields = line.split(' ')
        print(f"{fields[5]}\t{fields[6]}\t{float(fields[4])}")

def collect_count(filename: str):
    with open(filename, "r") as f:
        stats = json.load(f)[0]
        print(stats["count"])

def collect_greedy(filename: str):
    with open(filename, "r") as f:
        line = f.read().strip()
        fields = line.split(',')
        print(f"{fields[0]}\t{fields[1]}\t{fields[2]}")


if __name__ == "__main__":
    print("Name\tTime\tError\tCR")
    for f in glob.glob("/Users/zhgguo/Documents/projects/tensor_networks/output/BigEarthNet-v1_0_all/*/010/htucker_textfile/ht.txt"):
        name = f.split("/")[-4]
        print(name, end="\t")
        collect_ht(f)