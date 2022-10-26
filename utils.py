import json
from collections import Counter
import os


import numpy as np

INPUT_DIM = 128



def collision_probability(values):
    counter = Counter(values)
    values_num = len(values)
    return sum([(v * (v - 1)) for v in counter.values()]) / (values_num * (values_num - 1))

def collision_count(values):
    counter = Counter(values)
    return sum([v for v in counter.values() if v > 1]) / len(values)


def calc_collision_stats(values, group_num = 10):
    p_values = []
    for i in range(group_num):
        data = values[i * (len(values) // group_num): (i + 1) * (len(values) // group_num)]
        p = collision_probability(data) 
        p_values.append(p)
    # print(p_values)
    return np.mean(p_values), np.std(p_values), (np.std(p_values) / np.mean(p_values))   


class Experiemnt:
    #read experiment results from json file
    def __init__(self, results_folder):
        with open(os.path.join(results_folder, 'results.json'), 'r') as f:
            self.collision_probability = json.load(f)["collision_probability"]
        #load values from valus.json
        with open(os.path.join(results_folder, 'values.json'), 'r') as f:
            self.values = json.load(f)
            
        # load_params from file name
        self.num_nodes, self.edge_prob, self.network_size, self.seed, self.num_permutations = [float(x.split('=')[1]) for x in results_folder.split('/')[-1].split(',')]




if __name__ == '__main__':

    # load values from json file
    with open('values.json', 'r') as f:
        values = json.load(f)
    
    print("mean, std, std/mean")

    print(len(values))

    # calculate collision probability
    # print(collision_probability(values))

    print(f"{calc_collision_stats(values, 5)=}")

    print(f"{calc_collision_stats(values, 10)=}")

    # print(f"{calc_collision_stats(values, 25)=}")

    # print(f"{calc_collision_stats(values, 100)=}")


