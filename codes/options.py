import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=41, help='epoch number')
parser.add_argument('--lr', type=float, default=2.5e-6, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--tr_img_root', type=str, default='/home/yzhang1/PythonProjects/Pano_dataset/IMG/TRAIN_SOD/rgb/', help='')
parser.add_argument('--tr_gt_root', type=str, default='/home/yzhang1/PythonProjects/Pano_dataset/IMG/TRAIN_SOD/gt/', help='')
parser.add_argument('--save_path', type=str, default=os.getcwd() + '/PINet_cpts/', help='the path to save models and logs')
opt = parser.parse_args()
