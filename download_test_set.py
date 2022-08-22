'''
This source code is licensed under the license found in the LICENSE file.
This is the implementation of the "Learning to deblur using light field generated and real defocus images" paper accepted to CVPR 2022. 
Project GitHub repository: https://github.com/lingyanruan/DRBNet
Email: lyruanruan@gmail.com
Copyright (c) 2022-present, Lingyan Ruan
'''

## Download DPDD,RealDOF,CUHK,PixelDP test dataset
import os
import gdown
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--DPDD', action='store_true', help='download DPDD test set')
parser.add_argument('--RealDOF', action='store_true', help='download RealDOF test set')
parser.add_argument('--CUHK', action='store_true', help='download CUHK test set')
parser.add_argument('--PixelDP', action='store_true', help='download PixelDP test set')


args = parser.parse_args()

### Google drive IDs ######
dpdd_test  = '1W9HgltHkdQtLjEyhVEl4MTzxmYVGK2-3'  # https://drive.google.com/file/d/1W9HgltHkdQtLjEyhVEl4MTzxmYVGK2-3/view?usp=sharing
realdof_test  = '18MBe-b4txSMsMtPpPQ40YD4dhtJXCvyf' #https://drive.google.com/file/d/18MBe-b4txSMsMtPpPQ40YD4dhtJXCvyf/view?usp=sharing
cuhk_test  = '1HEUE5gIW35VwjLsxukk-fQ2KcvmAMtfC'   # https://drive.google.com/file/d/1HEUE5gIW35VwjLsxukk-fQ2KcvmAMtfC/view?usp=sharing
pixeldp_test  = '12K038LdCjfjLqR68v09nrmK6pWstibRV' #https://drive.google.com/file/d/12K038LdCjfjLqR68v09nrmK6pWstibRV/view?usp=sharing


#  download test dataset
if args.DPDD:
    print('DPDD Testing Data!')
    gdown.download(id=dpdd_test, output='datasets/DPDD.zip', quiet=False) 
    print('Extracting DPDD test set...')
    shutil.unpack_archive('datasets/DPDD.zip', 'datasets')
    os.remove('datasets/DPDD.zip')
    print('Successfully download DPDD!')

if args.RealDOF:
    print('RealDOF Testing Data!')
    gdown.download(id=realdof_test, output='datasets/RealDOF.zip', quiet=False)
    print('Extracting RealDOF test set...')
    shutil.unpack_archive('datasets/RealDOF.zip', 'datasets')
    os.remove('datasets/RealDOF.zip')
    print('Successfully download RealDOF!')

if args.CUHK:
    print('CUHK Testing Data!')
    gdown.download(id=cuhk_test, output='datasets/CUHK.zip', quiet=False) 
    print('Extracting CUHK test set...')
    shutil.unpack_archive('datasets/CUHK.zip', 'datasets')
    os.remove('datasets/CUHK.zip')
    print('Successfully download CUHK!')

if args.PixelDP:
    print('PixelDP Testing Data!')
    gdown.download(id=pixeldp_test, output='datasets/PixelDP.zip', quiet=False) 
    print('Extracting PixelDP test set...')
    shutil.unpack_archive('datasets/PixelDP.zip', 'datasets')
    os.remove('datasets/PixelDP.zip')
    print('Successfully download PixelDP!')

