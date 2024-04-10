
import time
from tensorboardX import SummaryWriter
import torch.optim as optim
from CTCSN import CTCSN
from dataset import *
# from tensorboardX import SummaryWriter
from trainOps import *
import os


# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 16
device = torch.device('cuda:0')
MAX_EP = 10000  # 最大轮次
VAL_HR = 256
INTERVAL = 4
WIDTH = 4
BANDS = 172
SIGMA = 0.0  ## Noise free -> SIGMA = 0.0
             ## Noise mode -> SIGMA > 0.0

SOURCE = '237C'
TARGET = '27C'
CR = 1  # compression ratio CR = 1, 5, 10, 15, 20
prefix = 'Train-C'
SNR = 25
LR = 0.0003


def trainer():
    traindata = loadTxt('trainpath/train_%s.txt' % SOURCE)
    testdata = loadTxt('testpath/val_%s.txt' % TARGET)

    train_loader = torch.utils.data.DataLoader(dataset_h5(traindata, width=WIDTH, marginal=60, root=''),
                                               batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset_h5(testdata, mode='Validation', root=''), batch_size=10,
                                             shuffle=False, pin_memory=False)

    model = CTCSN(snr=0, cr=CR).to(device)
    # model = torch.nn.DataParallel(model, device_ids=[0, 2, 3, 4])

    #state_dict = torch.load('checkpoint/Train-U_154U_18U_cr_20_epoch_200.pth')  ## finetune
    #model.load_state_dict(state_dict)                  ## finetune
    model.train()
    optimizer = torch.optim.RAdam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    L1Loss = torch.nn.L1Loss()

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('Rec'):
        os.mkdir('Rec')
    if not os.path.isdir('log'):
        os.mkdir('log')
    writer = SummaryWriter('log/%s_exp2_%s_%s_%s' % (prefix, SOURCE, TARGET, CR))

    resume_ind = 0
    step = resume_ind
    best_sam = 0.5

    for epoch in range(resume_ind, MAX_EP):
        print('epoch:', epoch)
        ep_loss = 0.
        for batch_idx, (x, _) in enumerate(train_loader):
            running_loss = 0.
            optimizer.zero_grad()
            x = x.view(x.size()[0] * x.size()[1], x.size()[2], x.size()[3], x.size()[4])
            x = x.to(device).permute(0, 3, 1, 2).float()
            decoded, _ = model(x)
            loss = L1Loss(decoded, x)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            with torch.no_grad():
                rmses, sams, fnames, psnrs = [], [], [], []
                start_time = time.time()
                for ind2, (vx, vfn) in enumerate(val_loader):
                    model.eval()
                    vx = vx.view(vx.size()[0] * vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
                    vx = vx.to(device).permute(0, 3, 1, 2).float()
                    if SIGMA > 0:
                        val_dec = model(awgn(model(vx, mode=1), SNR), mode=2)
                    else:
                        val_dec, _ = model(vx)

                    val_batch_size = len(vfn)
                    img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
                    val_dec = val_dec.permute(0, 2, 3, 1).cpu().numpy()
                    cnt = 0

                    for bt in range(val_batch_size):
                        for z in range(0, VAL_HR, INTERVAL):
                            img[bt][:, z:z + WIDTH, :] = val_dec[cnt]
                            cnt += 1
                        save_path = vfn[bt].split('/')
                        save_path = save_path[-1] + '-' + save_path[0]
                        np.save('Rec/%s.npy' % (save_path), img[bt])

                        GT = lmat(vfn[bt]).astype(np.float32)
                        maxv, minv = np.max(GT), np.min(GT)
                        img[bt] = img[bt] * (maxv - minv) + minv  ## De-normalization
                        sams.append(sam(GT, img[bt]))
                        rmses.append(rmse(GT, img[bt]))
                        fnames.append(save_path)
                        psnrs.append(psnr(img[bt], GT))

                ep = time.time() - start_time
                ep = ep / len(sams)
                plog(
                    '[epoch: %d, batch: %5d] loss: %.3f, , val-RMSE: %.3f, val-SAM: %.3f, val-PSNR: %.3f, AVG-Time: %.3f' %
                    (epoch, batch_idx + resume_ind, running_loss, np.mean(rmses), np.mean(sams), np.mean(psnrs), ep)
                    , prefix, SOURCE, TARGET, CR, epoch)
                writer.add_scalar('Validation RMSE', np.mean(rmses), step)
                writer.add_scalar('Validation SAM', np.mean(sams), step)
                scheduler.step(np.mean(psnrs))

                with open('log/%s_validataion_%s_%s_%d.txt' % (prefix, SOURCE, TARGET, CR), 'a') as fp:
                    if epoch == 0:
                        fp.write("\n")
                        fp.write("\n")
                        fp.write('EPOCH = {}, LR = {}, CR = {}, model = {}'.format(MAX_EP, LR, CR, prefix))
                    for p, r, s, f in zip(psnrs, rmses, sams, fnames):
                        fp.write("%d: %s:\tRMSE:%.4f\tSAM:%.3f\tPSNR:%.3f\n" % (epoch, f, r, s, p))
                    fp.write("----------------------------------------------------\n")

            if best_sam > np.mean(sams):
                best_sam = np.mean(sams)
                torch.save(model.state_dict(), 'checkpoint/%s_%s_%s_cr_%d_epoch_%d_%.3f.pth' %(prefix, SOURCE, TARGET, CR, epoch, np.mean(sams)))

            ep_loss += running_loss
            writer.add_scalar('Running loss', running_loss, step)

            running_loss = 0.0
            model.train()

        if epoch % 200 == 0 and epoch > 1:
            torch.save(model.state_dict(),
                       'checkpoint/%s_%s_%s_cr_%d_epoch_%d.pth' % (prefix, SOURCE, TARGET, CR, epoch))

        step += 1


if __name__ == '__main__':
    trainer()
