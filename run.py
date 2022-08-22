'''
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.
Note: this code is the implementation of the "Learning to deblur using light field generated and real defocus images" paper accepted to CVPR 2022. 
Project GitHub repository: https://github.com/lingyanruan/DRBNet
Email: lyruanruan@gmail.com
Copyright (c) 2022-present, Lingyan Ruan
'''

import os
from options.test_options import TestOptions
from datetime import datetime
import torch
import torchvision.utils as vutils
from ptflops import get_model_complexity_info
from util.util import *
from pathlib import Path
import time
import sys
import lpips
from glob import glob
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from models.DRBNet import *

#### metrics #################################
compute_lpips = lpips.LPIPS(net='alex').cuda()

opt = TestOptions().parse()

#### define time
folder_time = datetime.now().strftime('%Y-%m-%d_%H%M')

# results save position
opt.results_dir = opt.results_dir + '/' + opt.name + '/' + opt.eval_data + '/' + opt.net_mode +'/'+ folder_time

#### make directory ################################
Path(os.path.join(opt.results_dir, 'input' )).mkdir(parents=True, exist_ok=True)
Path(os.path.join(opt.results_dir, 'output')).mkdir(parents=True, exist_ok=True)

## evaluation values
PSNR_total,SSIM_total,LPIPS_total = 0,0,0
PSNR_score, SSIM_score, LPIPS_score , total_time= 0,0,0,0


######################################### Dataset List #################################################
input_c_file_path_list = []

if opt.eval_data == 'DPDD':
    input_c_file_path_list = natsorted(glob(os.path.join(opt.dataroot_dpdd, 'test_c','source', '*.png')))
    input_r_file_path_list = natsorted(glob(os.path.join(opt.dataroot_dpdd, 'test_r', 'source', '*.png')))
    input_l_file_path_list = natsorted(glob(os.path.join(opt.dataroot_dpdd, 'test_l', 'source','*.png')))
    gt_file_path_list = natsorted(glob(os.path.join(opt.dataroot_dpdd, 'test_c', 'target', '*.png')))

elif opt.eval_data == 'RealDOF':
    input_c_file_path_list = natsorted(glob(os.path.join(opt.dataroot_rf, 'source', '*.png')))
    gt_file_path_list = natsorted(glob(os.path.join(opt.dataroot_rf, 'target', '*.png')))    

elif opt.eval_data == 'PixelDP':
    input_c_file_path_list = natsorted(glob(os.path.join(opt.dataroot_pixeldp, 'test_c','source', '*.png')))
    gt_file_path_list = None

elif opt.eval_data == 'CUHK':
    input_c_file_path_list = natsorted(glob(os.path.join(opt.dataroot_cuhk,'*')))
    gt_file_path_list = None

total_files = len(input_c_file_path_list)

assert total_files > 0, 'Wrong Dataset Name or No Dataset Exist, Please Check!!'

print('\n\n================================= EVALUATION START ==================================================')

for i, filename in enumerate(input_c_file_path_list):
    # Read Image
    C = crop_image(read_image(input_c_file_path_list[i], 255.0))*2-1 
    C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).cuda()
    filename = os.path.split(filename)[-1]

    if opt.net_mode == 'dual':
        R,L = crop_image(read_image(input_r_file_path_list[i], 255.0))*2-1, crop_image(read_image(input_l_file_path_list[i], 255.0))*2-1
        R,L = torch.FloatTensor(R.transpose(0, 3, 1, 2).copy()).cuda(), torch.FloatTensor(L.transpose(0, 3, 1, 2).copy()).cuda()
    if gt_file_path_list is not None:
        GT = crop_image(read_image(gt_file_path_list[i], 255.0))  # here to [0,1]
        GT = torch.FloatTensor(GT.transpose(0, 3, 1, 2).copy()).cuda()

    ##test resut
    with torch.no_grad():
        
        if opt.net_mode == 'single':
            network = DRBNet_single().cuda()
            opt.ckpt_path = './ckpts/single/single_image_defocus_deblurring.pth' #final one
            network.load_state_dict(torch.load(opt.ckpt_path))
            start_time = time.time()
            output = network(C)
            time_per = time.time() - start_time
        else:
            network = DeblurNet_dual().cuda()
            opt.ckpt_path = './ckpts/dual/dual_images_defocus_deblurring.pth'
            network.load_state_dict(torch.load(opt.ckpt_path))
            start_time = time.time()
            output = network(C,R,L)
            time_per = time.time() - start_time
             
                
        total_time = total_time + time_per

    output_cpu = (output.cpu().numpy()[0].transpose(1, 2, 0) +1.0 )/2.0 # to [0,1] for psnr and ssim evaluation
        
    if gt_file_path_list is not None:
        GT_cpu = GT.cpu().numpy()[0].transpose(1, 2, 0) 
        PSNR_score = compute_psnr(output_cpu, GT_cpu,data_range=1.0)
        SSIM_score = compute_ssim(output_cpu, GT_cpu,data_range=1.0,multichannel=True)
        LPIPS_score = compute_lpips(output, GT * 2. - 1.).item()

    if opt.save_images:
        save_file_path_deblur_input = os.path.join(opt.results_dir, 'input',  '{}'.format(filename))
        save_file_path_deblur = os.path.join(opt.results_dir, 'output', '{}'.format(filename))
        vutils.save_image((C+1.0)/2.0, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)
        vutils.save_image((output+1.0)/2.0, '{}'.format(save_file_path_deblur), nrow=1, padding = 0, normalize = False)

    # Log
    print('[EVAL on {}][{:02}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f}, Time: {:.5f}sec'.format( opt.eval_data, i + 1, total_files, filename, PSNR_score, SSIM_score,  LPIPS_score, time_per))
    with open(os.path.join(opt.results_dir, 'score_{}.txt'.format(opt.eval_data)), 'w' if i == 0 else 'a') as file:
        file.write('[EVAL][{:02}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f}, Time: {:.5f}sec \n'.format( i + 1, total_files, filename, PSNR_score, SSIM_score,  LPIPS_score, time_per))
        file.close()

    PSNR_total += PSNR_score
    SSIM_total += SSIM_score
    LPIPS_total += LPIPS_score

###=============================== network parameters info =======================================#######
PSNR_mean,SSIM_mean,LPIPS_mean,time_mean = PSNR_total / total_files,SSIM_total / total_files, LPIPS_total/total_files, total_time/total_files

def prepare_input(resolution):
    input_blur_C = torch.FloatTensor(1, 3, 720, 1280).cuda()
    input_blur_L = torch.FloatTensor(1, 3, 720, 1280).cuda()
    input_blur_R = torch.FloatTensor(1, 3, 720, 1280).cuda()
    return dict(C = input_blur_C, R=input_blur_L, L=input_blur_R)


### add network parameters info#######
if opt.net_mode == 'single':
    Macs,params = get_model_complexity_info(network, (3, 720, 1280), as_strings=False)
    print('\t{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
    print('\t{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2, '\n'))

else:
    Macs,params = get_model_complexity_info(network, (1,3, 720, 1280),input_constructor=prepare_input, as_strings=False,print_per_layer_stat=False)
    print('\t{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
    print('\t{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2, '\n'))


sys.stdout.write('\n[TOTAL |{}] PSNR: {:.5f} SSIM: {:.5f} LPIPS: {:.5f} ({:.5f}sec)'.format(opt.eval_data, PSNR_mean, SSIM_mean,  LPIPS_mean, time_mean))
with open(os.path.join(opt.results_dir, 'score_{}.txt'.format(opt.eval_data)), 'a') as file:
    file.write('\n[TOTAL ] PSNR: {:.5f} SSIM: {:.5f}  LPIPS: {:.5f} ({:.5f}sec)'.format( PSNR_mean, SSIM_mean,  LPIPS_mean, time_mean))
    file.write('\n{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
    file.write('\n{:<30}  {:<8} M'.format('Number of parameters: ', params / 1000 ** 2, '\n'))
    file.close()







