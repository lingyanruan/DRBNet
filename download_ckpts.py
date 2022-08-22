'''
This source code is licensed under the license found in the LICENSE file.
This is the implementation of the "Learning to deblur using light field generated and real defocus images" paper accepted to CVPR 2022. 
Project GitHub repository: https://github.com/lingyanruan/DRBNet
Email: lyruanruan@gmail.com
Copyright (c) 2022-present, Lingyan Ruan
'''

## Download weight ##############
import os
import gdown
import shutil

### Google drive IDs ######
ckpt_test  = '1vGImev9LdagttXE_nN1gZGVstVTRVQHt'  # https://drive.google.com/file/d/1vGImev9LdagttXE_nN1gZGVstVTRVQHt/view?usp=sharing

#  download ckpts
print('ckpt downloading!')
gdown.download(id=ckpt_test, output='ckpts/ckpts.zip', quiet=False) 
print('Extracting ckpts ......')
shutil.unpack_archive('ckpts/ckpts.zip')
os.remove('ckpts/ckpts.zip')
print('Successfully download weight!')


