# ==============================================================================
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper
# All rights reserved.
# ==============================================================================

import os
import subprocess

yn = str(input(
    '\nBy participating in the MOS Prediction Challenge and downloading this Blizzard Challenge data, you agree to abide by the terms of use of this data.  This data may NOT be redistributed.  For more information, see https://www.cstr.ed.ac.uk/projects/blizzard/data.html. \n Do you agree?  y/n > '))

if yn not in ['y', 'Y', 'yes']:
    print('You must agree to the terms to download the data and participate in the challenge.  Exiting.')
    exit()

print(' ===== Starting download of Blizzard samples ===== \n')

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2008_release_version_1.tar.bz2'
print(cmd)
os.system(cmd)

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2009_release_version_1.tar.bz2'
print(cmd)
os.system(cmd)

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2010_release_version_1.tar.bz2'
print(cmd)
os.system(cmd)

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2011_release_version_1.tar.bz2'
print(cmd)
os.system(cmd)

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2013_release_version_2.tar.bz2'
print(cmd)
os.system(cmd)

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2016_release_version_1.tar.bz2'
print(cmd)
os.system(cmd)

## next: check checksums
print(' ===== Checking checksums ===== \n')

sums = {
    '2008': '5360686aac07ffe22420ffd00c90ea74',
    '2009': '1ffdf2c0ddb5f2e0c97908a70d0302b2',
    '2010': 'e60d504c4e3a95d7792e6fd056e74aec',
    '2011': '8e59a48f88568f86d1962644fdd568c5',
    '2013': '837e20399409393332322fdd59d114de',
    '2016': 'fff9d42a97161835f2545e02e5392e06'
}

for year in ['2008', '2009', '2010', '2011', '2013', '2016']:
    ver = '1'
    if year == '2013':
        ver = '2'
    result = subprocess.run(
        ['md5sum', 'blizzard/blizzard_wavs_and_scores_' + year + '_release_version_' + ver + '.tar.bz2'],
        stdout=subprocess.PIPE)
    checksum = result.stdout.decode('utf-8').split()[0]
    if checksum != sums[year]:
        print(
            'UH OH: file for ' + year + ' may be corrupted.  Please delete the corrupted file and retry downloading it.')
        exit()
    else:
        print('File for ' + year + ' downloaded ok')

print(' ===== Extracting data ===== \n')

for year in ['2008', '2009', '2010', '2011', '2013', '2016']:
    ver = '1'
    if year == '2013':
        ver = '2'
    cmd = 'tar -xf blizzard/blizzard_wavs_and_scores_' + year + '_release_version_' + ver + '.tar.bz2 --directory blizzard'
    if not os.path.exists('blizzard/blizzard_wavs_and_scores_' + year + '_release_version_' + ver):
        print(cmd)
        os.system(cmd)
    else:
        print(year + ' already extracted')

## fix permission problem for BC2009 system F
cmd = 'chmod -R u+rwX,go+rX,go-w blizzard/blizzard_wavs_and_scores_2009_release_version_1/F/submission_directory/'
# print(cmd)
os.system(cmd)

## fix permission problem for BC2010 system J
cmd = 'chmod u+w blizzard/blizzard_wavs_and_scores_2010_release_version_1/J/submission_directory/english/EH1/2010/*/wavs/*'
# print(cmd)
os.system(cmd)

print(' ===== Cleaning up ===== \n')
# cmd = 'rm blizzard/*.tar.bz2'
# print(cmd)
# os.system(cmd)

os.system('rm blizzard/*/*.csv')
os.system('rm -rf blizzard/*/statistics')
os.system('rm -rf blizzard/*/main_test_results_files')
os.system('rm -rf blizzard/*/mechanical_turk_results_files')
os.system('rm -rf blizzard/*/test_results')

print(' ===== Done ===== \n')
