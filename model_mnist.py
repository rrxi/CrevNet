import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_utils
import numpy as np
import time
import pssim.pytorch_ssim as pytorch_ssim
from skimage.measure import compare_ssim
from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=5000, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=8, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--n_eval', type=int, default=18, help='number of frames to predict at eval time')
parser.add_argument('--rnn_size', type=int, default=32, help='dimensionality of hidden layer')
parser.add_argument('--predictor_rnn_layers', type=int, default=8, help='number of layers')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='crevnet', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')



opt = parser.parse_args()




if opt.model_dir != '':
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model_mnist=layers_%s=seq_len_%s=batch_size_%s' % (opt.predictor_rnn_layers,opt.n_eval,opt.batch_size)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)





opt.max_step = opt.n_past + opt.n_future + 2
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor
opt.data_type = 'sequence'
# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
opt.optimizer = optim.Adam


import layers_3d as model


frame_predictor = model.zig_rev_predictor(opt.rnn_size,  opt.rnn_size, opt.rnn_size, opt.predictor_rnn_layers,opt.batch_size)
encoder = model.autoencoder(nBlocks=[4,5,3], nStrides=[1, 2, 2],
                    nChannels=None, init_ds=2,
                    dropout_rate=0., affineBN=True, in_shape=[opt.channels, opt.image_width, opt.image_width],
                    mult=2)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


scheduler1 = torch.optim.lr_scheduler.StepLR(frame_predictor_optimizer, step_size=50, gamma=0.2)
scheduler2 = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=50, gamma=0.2)


# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()


# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
encoder.cuda()
mse_criterion.cuda()


# --------- load a dataset ------------------------------------
train_data, test_data = data_utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=False)


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = data_utils.normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = data_utils.normalize_data(opt, dtype, sequence)
            yield batch


testing_batch_generator = get_testing_batch()

def plot(x, epoch,p = False):
    nsample = 1
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]
    mse = 0
    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size ,3, int(opt.image_width/8), int(opt.image_width/8)).cuda())
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if i < opt.n_past:
                _,memo = frame_predictor((h,memo))
                x_in = x[i]
                gen_seq[s].append(x_in)
            elif i == opt.n_past:
                h_pred, memo = frame_predictor((h, memo))
                x_in = encoder(h_pred, False).detach()
                x_in[:, :, 0] = x[i][:, :, 0]
                x_in[:, :, 1] = x[i][:, :, 1]
                gen_seq[s].append(x_in)
            elif i == opt.n_past + 1:
                h_pred, memo = frame_predictor((h, memo))
                x_in = encoder(h_pred, False).detach()
                x_in[:, :, 0] = x[i][:, :, 0]
                gen_seq[s].append(x_in)
            else:
                h_pred, memo = frame_predictor((h, memo))
                x_in =encoder(h_pred,False).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [[] for t in range(opt.n_eval)]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i][0][2])
        to_plot.append(row)
        mse = 0
        for s in range(nsample):
            for t in range(opt.n_past,opt.n_eval):
                mse += pytorch_ssim.ssim(gt_seq[t][:,0,2][:,None, :, :],gen_seq[s][t][:,0,2][:,None, :, :])
        s_list = [0]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i][0][2])
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i][0][2])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i][0][2])
            gifs[t].append(row)

    if p:
        fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch)
        data_utils.save_tensors_image(fname, to_plot)

        fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch)
        data_utils.save_gif(fname, gifs)
    return mse / 10.0


# --------- training funtions ------------------------------------
def train(x,e):
    frame_predictor.zero_grad()
    encoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    mse = 0

    memo = Variable(torch.zeros(opt.batch_size, opt.rnn_size ,3, int(opt.image_width/8), int(opt.image_width/8)).cuda())
    for i in range(1, opt.n_past + opt.n_future):
        h = encoder(x[i - 1], True)
        h_pred,memo = frame_predictor((h,memo))
        x_pred = encoder(h_pred,False)
        mse +=  (mse_criterion(x_pred, x[i]))


    loss = mse
    loss.backward()

    frame_predictor_optimizer.step()
    encoder_optimizer.step()

    return mse.data.cpu().numpy() / (opt.n_past + opt.n_future)



# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    encoder.train()
    epoch_mse = 0

    for i in trange(opt.epoch_size):
        x = next(training_batch_generator)
        input = []
        for j in range(opt.n_eval):
            k1 = x[j][:, 0][:,None,None,:,:]
            k2 = x[j + 1][:, 0][:,None,None,:,:]
            k3 = x[j + 2][:, 0][:,None,None,:,:]

            input.append(torch.cat((k1,k2,k3),2))
        mse = 0
        mse = train(input,epoch)
        epoch_mse += mse

    scheduler1.step()
    scheduler2.step()

    with torch.no_grad():
        frame_predictor.eval()
        encoder.eval()

        eval = 0
        for i in range(100):
            x = next(testing_batch_generator)
            input = []
            for j in range(opt.n_eval):
                k1 = x[j][:, 0][:, None, None, :, :]
                k2 = x[j + 1][:, 0][:, None, None, :, :]
                k3 = x[j + 2][:, 0][:, None, None, :, :]

                input.append(torch.cat((k1, k2, k3), 2))
            if i == 0:
                ssim = plot(input, epoch, True)
            else:
                ssim = plot(input, epoch)
            eval += ssim

        print('[%02d] mse loss: %.7f| ssim loss: %.7f(%d)' % (
            epoch, epoch_mse / opt.epoch_size,eval/ 100.0, epoch * opt.epoch_size * opt.batch_size))

    # save the model



    if epoch % 10 == 0:
        torch.save({
            'encoder': encoder,
            'frame_predictor': frame_predictor,
            'opt': opt},
            '%s/model_%s.pth' % (opt.log_dir,epoch))


    torch.save({
        'encoder': encoder,
        'frame_predictor': frame_predictor,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)




