import sys
import pickle
import pprint


pp = pprint.PrettyPrinter(indent=4)

file_to_unpickle = sys.argv[1]

with open(file_to_unpickle, 'rb') as target:
    obj = pickle.load(target)
    pp.pprint(obj)
