import argparse
import os
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        self.parser.add_argument('--dataroot_rf', default='./datasets/RealDOF', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--dataroot_pixeldp', default='./datasets/PixelDP', help='PixelDP dataset path')
        self.parser.add_argument('--dataroot_lf', default='./datasets/LFDOF/test_data', help='LFDOF dataset path')
        self.parser.add_argument('--dataroot_dpdd', default='./datasets/DPDD', help='DPDD dataset path')       
        self.parser.add_argument('--dataroot_cuhk', default='./datasets/CUHK', help='CUHK dataset path')       
        self.parser.add_argument('--name', type=str, default='defocus_deblur', help='name of the experiment. It decides where to store samples and models')
        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
