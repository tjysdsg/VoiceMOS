# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import sys

if len(sys.argv) != 2:
    print('Usage: python setup_main.py /path/to/your/phase1-main')
    exit()

datapath = sys.argv[1]

cmd = 'cp main/wav/* ' + os.path.join(datapath, 'DATA/wav/')
os.system(cmd)

cmd = 'cp main/.scp.zip ' + datapath
os.system(cmd)

cmd = 'cp main/*.py ' + datapath
os.system(cmd)

cmd = 'cp main/test.scp ' + os.path.join(datapath, 'DATA/sets/')
os.system(cmd)
