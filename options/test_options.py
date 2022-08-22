from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--eval_data', type=str, default='DPD', help='DPD|LF|RealDOF|PixelDP')
        self.parser.add_argument('--save_images', action='store_true', help='save images')
        self.parser.add_argument('--net_mode', type=str, default='single', help='single | dual')
        self.parser.add_argument('--ckpt_path', type=str, default='./ckpts/', help='single | dual')
        self.isTrain = False