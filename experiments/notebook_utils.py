import sys
import pickle
import pprint
import numpy as np

pp = pprint.PrettyPrinter(indent=4)


def load_pickle(job_name, fn):
    with open(f'{job_name}/binary/{fn}', 'rb') as target:
        obj = pickle.load(target)
        return obj

def get_max_tile_frequency(max_tiles_array):
    max_tile_frequency = np.array(np.unique(max_tiles_array, return_counts=True), dtype=int)
    return max_tile_frequency

def plot_max_tile_distribution(max_tile_frequency, ax, include_labels=True):
    barwidth= 0.8
    ax.bar(np.arange(len(max_tile_frequency[0])) , max_tile_frequency[1], barwidth, color="pink")
    ax.set_xticks(np.arange(len(max_tile_frequency[1]))+barwidth/2.)
    ax.set_xticklabels(max_tile_frequency[0])
    if include_labels:
        ax.set_xlabel("Tiles Reached")
        ax.set_ylabel("Frequency")

if __name__ == "__main__":
    job_name = sys.argv[1]
    filename = sys.argv[2]


    obj = load_pickle(job_name=job_name, fn=filename)
    pp.pprint(obj)
