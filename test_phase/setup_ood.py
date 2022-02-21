# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import sys

if len(sys.argv) != 2:
    print('Usage: python setup_ood.py /path/to/your/phase1-ood')
    exit()

datapath = sys.argv[1]

cmd = 'cp ood/.scp.zip ' + datapath
os.system(cmd)

cmd = 'cp ood/*.py ' + datapath
os.system(cmd)

cmd = 'cp ood/test.scp ' + os.path.join(datapath, 'DATA/sets/')
os.system(cmd)
