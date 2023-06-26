from __future__ import division
import numpy as np
import generator_pure_dict as gen
import os
import math
import sys
import getopt


def main(argv):
    msets = 100
    processors = 16
    pc_prob = 0
    group_mode = 0
    sparse = 0
    group_prob = 0
    scale = 10**6
    util = 5
    try:
        opts, args = getopt.getopt(argv, "hm:p:q:s:g:r:u:",
                                   ["msets=", "processors", "pc_prob=", "sparse=", "gpm=", "gpq=", "util"])
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
        elif opt in ("-u", "--util"):
            util = int(arg)

    utili = float(util / 100)
    utilization = utili*processors
    tasksets_name = '../experiments/inputs/tasksets_m' + str(msets) + '_p' + str(processors) + '_u' + str(utili) + '_q' + str(pc_prob)+ '_s' + str(sparse)+ '_g' + str(group_mode)+ '_r' + str(group_prob)+'.npy'
    tasksets = gen.generate_tsk_dict(msets, processors, pc_prob, utilization, sparse, group_mode, group_prob, scale)
    np.save(tasksets_name, tasksets)

if __name__ == "__main__":
    main(sys.argv[1:])
