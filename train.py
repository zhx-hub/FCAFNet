import os
import torch
import torch.nn.functional as F
import sys
import numpy as np
from datetime import datetime
from models.FCAFNet import Net
from torchvision.utils import make_grid
from data import SalObjDataset,TestDataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt



if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True


save_path = opt.save_path
# load data
print('load data...')

dataset_train = SalObjDataset('./train_2985/',384 )
dataset_val = TestDataset('./DUT-RGBD/',384 )
train_loader = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size=opt.batchsize,
    num_workers=opt.num_workers,
    pin_memory=True,
    shuffle=True,

)

test_loader = torch.utils.data.DataLoader(
    dataset_val, 
    batch_size=int(1.5 * opt.batchsize),
    num_workers=opt.num_workers,
    pin_memory=True,
    shuffle=True,

)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'FCAFNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("FCAFNet-Train")

# build the model
model = Net(ckpt=opt.backbonet_path,img_size=(384,384))
model.init_weights()

num_parms = 0
if (opt.load_pre is not None):
    model.load_pre(opt.load_pre)
    print('load model from ', opt.load_pre)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load_pre, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
model=model.to('cuda')

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth, edge,_) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depth = depth.cuda()
            edge = edge.cuda()
            s,e = model(images, depth)

            sal_loss = CE(s, gts)
            edge_loss = CE(e, edge)
            loss = sal_loss + edge_loss
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} ||edge_loss:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data, edge_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||edge_loss:{:4f} , mem_use:{:.0f}MB'.
                    format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                           sal_loss.data, edge_loss.data, sal_loss.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 20== 0 or epoch>=290:
            torch.save(model.state_dict(), save_path + 'CATNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'CATNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise
    return loss_all

# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        #for i in range(test_loader.size):
        for i, (images, gts, depth, edge,_) in enumerate(test_loader, start=1):
            image, gt, depth,  = images, gts, depth,#test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res,e = model(image, depth)
            #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / len(dataset_val)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'CATNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    temp1=10
    for epoch in range(1, opt.epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        loss_all=train(train_loader, model, optimizer, epoch, save_path)
        if loss_all<temp1:
            temp1=loss_all
            print('Loss_save:{} Epoch [{:03d}], ||total_loss:{:4f}'.
                    format(datetime.now(), epoch, 
                            temp1))
            torch.save(model.state_dict(), save_path + 'FCAFNet_epoch_total_loss_all.pth') 
        test(test_loader, model, epoch, save_path)















