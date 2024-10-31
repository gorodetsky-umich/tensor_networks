import glob
import json

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

if __name__ == "__main__":
    print("Name\tTime\tError\tCR")
    for f in glob.glob("output/BigEarthNet-v1_0_all/*/010/*/ht.txt"):
        name = f.split("/")[2]
        print(name, end="\t")
        collect_ht(f)