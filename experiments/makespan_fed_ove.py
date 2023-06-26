from __future__ import division
import numpy as np
import os
import math
import sys
import getopt
sys.path.append('../')
from generators import generator_pure_dict
from algorithms import graph
from algorithms import Federated as fed

def main(argv):
    msets = 10
    processors = 16
    pc_prob = 0
    group_mode = 0
    sparse = 0
    group_prob = 0
    over = 1
    scale = 10**6

    try:
        opts, args = getopt.getopt(argv, "hm:p:q:s:g:r:o:",
                                   ["msets=", "processors", "pc_prob=", "sparse=", "gpm=", "gpq=", "ove="])
    except getopt.GetoptError:
        print ('tasksets_generater.py -n <n tasks for each set> -m <m tasksets> -p <num of processors>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('tasksets_generater.py -n <n tasks for each set> -m <m tasksets> -p <num of processors>')
            sys.exit()
        elif opt in ("-m", "--msets"):
            msets = int(arg)
        elif opt in ("-p", "--processors"):
            processors = int(arg)
        elif opt in ("-q", "--pc_prob"):
            pc_prob = int(arg)
        elif opt in ("-s", "--sparse"):
            sparse = int(arg)
        elif opt in ("-g", "--gpm"):
            group_mode = int(arg)
        elif opt in ("-r", "--gpq"):
            group_prob = int(arg)
        elif opt in ("-o", "--ove"):
            over = int(arg)

    makespan_all = []

    for util in range(5, 55, 5):
        utili = float(util / 100)
        overheads = 1 + over / 10
        tasksets_name = './inputs2/tasksets_m' + str(msets) + '_p' + str(processors) + '_u' + str(
            utili) + '_q' + str(pc_prob) + '_s' + str(sparse) + '_g' + str(group_mode) + '_r' + str(group_prob) + '_o' + str(overheads) + '.npy'
        tasksets = np.load(tasksets_name, allow_pickle = True)

        makespan_all.append(fed.calculate_makespan_all(tasksets, processors))

    results_name = './outputs2/makespan_fed_m' + str(msets) + '_p' + str(processors) + '_q' + str(pc_prob)+ '_s' + str(sparse) + '_g' + str(group_mode) + '_r' + str(group_prob) + '_o' + str(over)+'.npy'
    np.save(results_name, makespan_all)


if __name__ == "__main__":
    main(sys.argv[1:])
