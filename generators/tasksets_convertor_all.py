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

    try:
        opts, args = getopt.getopt(argv, "hm:p:q:s:g:r:",
                                   ["msets=", "processors", "pc_prob=", "sparse=", "gpm=", "gpq="])
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

    if group_mode == 0 and group_prob == 0:
        return

    for util in range(5, 55, 5):
        utili = float(util / 100)
        utilization = utili*processors
        tasksets_input_name = '../experiments/inputs/tasksets_m' + str(msets) + '_p' + str(processors) + '_u' + str(utili) + '_q' + str(pc_prob)+ '_s' + str(sparse)+ '_g0_r0.npy'
        tasksets_output_name = '../experiments/inputs/tasksets_m' + str(msets) + '_p' + str(processors) + '_u' + str(
            utili) + '_q' + str(pc_prob) + '_s' + str(sparse) + '_g' + str(group_mode) + '_r' + str(group_prob) + '.npy'
        tasksets_input = np.load(tasksets_input_name, allow_pickle = True)
        tasksets_output = gen.tsk_dict_convertor(msets, processors, group_mode, group_prob, tasksets_input)
        np.save(tasksets_output_name, tasksets_output)

if __name__ == "__main__":
    main(sys.argv[1:])
