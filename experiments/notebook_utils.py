import sys
import pickle
import pprint


pp = pprint.PrettyPrinter(indent=4)


def load_pickle(job_name, fn):
    with open(f'{job_name}/binary/{fn}', 'rb') as target:
        obj = pickle.load(target)
        return obj

if __name__ == "__main__":
    job_name = sys.argv[1]
    filename = sys.argv[2]


    obj = load_pickle(job_name=job_name, fn=filename)
    pp.pprint(obj)
