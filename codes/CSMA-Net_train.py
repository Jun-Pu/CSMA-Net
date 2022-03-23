import os
import torch
import numpy as np
from datetime import datetime
from data import get_loader
from utils import clip_gradient
import logging
import torch.backends.cudnn as cudnn
from options import opt
from utils import print_network
from utils import structure_loss
import cv2

from models.CSMANet import ImgModel

#set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

#build the model
model = ImgModel()
print_network(model, 'CSMA-Net')
if(opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

#set the path
tr_img_root = opt.tr_img_root
tr_gt_root = opt.tr_gt_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(tr_img_root, tr_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CSMA-Net-Train")
logging.info("Config")
logging.info('epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};load:{};save_path:{}'.
             format(opt.epoch,opt.lr,opt.batchsize,opt.trainsize,opt.clip,opt.load,save_path))

step = 0
best_mae = 1
best_epoch = 0

#train function
def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, er_images, er_gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            er_images = er_images.cuda()
            er_gts = er_gts.cuda()

            # debug
            #cv2.imwrite('imag_1.png', images[0].permute(1, 2, 0).cpu().data.numpy() * 255)
            #cv2.imwrite('image_2.png', images[1].permute(1, 2, 0).cpu().data.numpy() * 255)
            #cv2.imwrite('image_1_er.png', er_images[0].permute(1, 2, 0).cpu().data.numpy() * 255)
            #cv2.imwrite('image_2_er.png', er_images[1].permute(1, 2, 0).cpu().data.numpy() * 255)
            #cv2.imwrite('gt_1.png', gts[0].permute(1, 2, 0).cpu().data.numpy() * 255)
            #cv2.imwrite('gt_2.png', gts[1].permute(1, 2, 0).cpu().data.numpy() * 255)

            preds, preds_aux, cube_gts = model(images, er_images, er_gts)
            loss = structure_loss(preds, gts) + structure_loss(preds_aux, cube_gts)

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 200 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data))
        
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), save_path+'CSMA-Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'CSMA-Net_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path)
