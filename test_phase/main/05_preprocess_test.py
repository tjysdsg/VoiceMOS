# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os

print('\n ===== Checking dependencies ===== \n')

gathered = os.listdir('gathered')
if len(gathered) != 325:
    print("*** ERROR *** There should be 325 files in the 'gathered' directory but there are " + str(len(gathered)) + '.  Did you run step 04?  Did it show any errors?  Please address any errors in the previous steps and then re-run this script.\n')
    exit()

if os.system('which sox') != 0:
    print('*** ERROR *** sox not found on your path.\nDo you have sox installed?\nIf not, please get it from here: http://sox.sourceforge.net\nThen, make sure it can be found on your system path.\nTo check if sox is on your system path, run \'which sox\'.\n')
    exit()

if os.system('which sv56demo') != 0:
    print('*** ERROR *** sv56demo not found on your path.\nPlease follow the instructions in the README to compile the sv56demo binary and add it to your system path.\nTo check if sv56demo is on your system path, run \'which sv56demo\'.\nPlease run this script again once you have sv56demo on your path.\n')
    exit()

print('All ok')

keeplst = [x.strip() for x in open('DATA/sets/test.scp', 'r').readlines()]

print('\n ===== Downsampling ===== \n')

cmd = 'mkdir -p downsampled'
os.system(cmd)
print('This takes a while, please wait....')

for w in os.listdir('gathered'):
    cmd = 'sox gathered/' + w + ' -r 16000 -V1 downsampled/' + w
    os.system(cmd)

print('Checking....')
ds = os.listdir('downsampled')
if len(ds) != 325:
    print("*** ERROR *** There should be 325 files in the 'downsampled' directory but there are " + str(len(ds)) + '.  Please address any errors in the previous step and then re-run this script.\n')
    exit()

print('\n ===== Normalizing ===== \n')

print('This takes a while, please wait....')

## normalize
sv56script = 'sv56scripts/batch_normRMSE.sh'
cmd = sv56script + ' downsampled'
os.system(cmd)

## sort and rename
cmd = 'mkdir -p normalized'
print(cmd)
os.system(cmd)
wavs = [x for x in os.listdir('downsampled') if x.split('_')[-1] == 'norm.wav']

if len(wavs) != 325:
    print("*** ERROR *** There should be 325 *_norm.wav files in the 'downsampled' directory but there are " + str(len(wavs)) + '.\nThis indicates that normalization did not run.\nDid you compile sv56demo and place it on your path?\nPlease follow the instructions in the README to do this and then re-run this script.\n')
    exit()

for w in wavs:
    outname = '_'.join(w.split('_')[0:-1]) + '.wav'
    cmd = 'mv downsampled/' + w + ' ' + 'normalized/' + outname
    os.system(cmd)

## copy all to DATA
cmd = 'cp normalized/* DATA/wav/'
print(cmd)
os.system(cmd)

print('\n ===== Sanity check ===== \n')

print('Checking.....')

missing_count = 0
for k in keeplst:
    if not os.path.isfile('DATA/wav/' + k):
        if missing_count < 5:
            print('MISSING: ' + k)
        elif missing_count == 5:
            print('....')
        missing_count += 1        

if missing_count > 0:
    print('*** ERROR *** There are missing files in your dataset.  Something went wrong.\nPlease address any errors and then re-run this script.\nYou have ' + str(missing_count) + ' missing files.\n')
    exit()
else:
    print('All data is present, you\'re all set for the main track evaluation phase of the challenge!  Good luck!')

print('\n ===== Cleaning up ===== \n')
    
cmd = 'rm -rf gathered downsampled normalized'
print(cmd)
os.system(cmd)

print('\n ===== Done ===== \n')
