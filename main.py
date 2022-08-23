import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

import params
from dataloader.read_data import *
import torch.optim as optim

from utils.python_pfm import *
from utils.util import *
from viewer.view import *
from models.myNet import *

p = params.Params()

head = HeadPac().to(p.device)
body = BodyFst().to(p.device)

criterion = HingeLoss(margin=0.2, reduction='mean',
                      ignore_pad=True, pad_value=0).to(p.device)
params = list(head.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)  # TODO: body 파라미터, 가변 lr
# optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9) # for mc-cnn

transform = transforms.Compose([transforms.ToTensor()])
#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# transforms.Grayscale(num_output_channels=1) # grayscale로 변환한다.


def train(imgL, imgR, imgL_guide, imgR_guide, disp_true, guide=False):
    head.train()

    optimizer.zero_grad()

    # with torch.autograd.set_detect_anomaly(True):
    if guide:
        featuresL, kernelsL = head(imgL.to(p.device), imgL_guide.to(p.device))
        featuresR, kernelsR = head(imgR.to(p.device), imgR_guide.to(p.device))
    else:
        featuresL, kernelsL = head(imgL.to(p.device))
        featuresR, kernelsR = head(imgR.to(p.device))
    cost_vols = body(featuresL, featuresR, p.train_disparity, kernelsL)

    gt = disp_true.to(p.device)
    gt[gt < 0] = float('inf')  # debug
    masks = []
    disps = []
    for i in range(len(cost_vols)):
        disp = (F.interpolate(gt, scale_factor=1/2**i,
                              mode='bilinear', align_corners=False))
        mask = disp < p.train_disparity - 0.5
        disp[mask == 0] = 0
        masks.append(mask)
        disp = ((disp + 0.5) // 2**i).long()  # (N, 1, H, W)
        disp = one_hot(disp, p.train_disparity // 2**i, dim=1)
        disps.append(disp)

    losses = []
    for cost_vol, disp, mask in zip(cost_vols, disps, masks):
        losses.append(criterion(cost_vol, disp, mask))
    loss = sum(losses)

    # with torch.autograd.detect_anomaly():
    loss.backward()

    optimizer.step()

    return [loss.cpu().detach().numpy() for loss in losses]


def validate(imgL, imgR, imgL_guide, imgR_guide, disp_true, guide=False):
    head.eval()

    optimizer.zero_grad()

    with torch.no_grad():
        if guide:
            featuresL, kernelsL = head(
                imgL.to(p.device), imgL_guide.to(p.device))
            featuresR, kernelsR = head(
                imgR.to(p.device), imgR_guide.to(p.device))
        else:
            featuresL, kernelsL = head(imgL.to(p.device))
            featuresR, kernelsR = head(imgR.to(p.device))
        cost_vols = body(featuresL, featuresR, p.train_disparity, kernelsL)

    gt = disp_true.to(p.device)
    masks = []
    disps = []
    for i in range(len(cost_vols)):
        disp = (F.interpolate(gt, scale_factor=1/2**i,
                              mode='bilinear', align_corners=False))
        mask = disp < p.train_disparity - 0.5
        disp[mask == 0] = 0
        masks.append(mask)
        disp = ((disp + 0.5) // 2**i).long()  # (N, 1, H, W)
        disp = one_hot(disp, p.train_disparity // 2**i, dim=1)
        disps.append(disp)

    losses = []
    for cost_vol, disp, mask in zip(cost_vols, disps, masks):
        losses.append(criterion(cost_vol, disp, mask))

    return [loss.cpu().numpy() for loss in losses]


def test(imgL, imgR, imgL_guide, imgR_guide, disp_true, guide=True):
    head.eval()

    optimizer.zero_grad()

    with torch.no_grad():
        if guide:
            featuresL, kernelsL = head(
                imgL.to(p.device), imgL_guide.to(p.device))
            featuresR, kernelsR = head(
                imgR.to(p.device), imgR_guide.to(p.device))
        else:
            featuresL, kernelsL = head(imgL.to(p.device))
            featuresR, kernelsR = head(imgR.to(p.device))
        cost_vols = body(featuresL, featuresR, p.test_disparity, kernelsL)

    gt = disp_true.to(p.device)
    masks = []
    disps = []
    for i in range(len(cost_vols)):
        disp = (F.interpolate(gt, scale_factor=1/2**i,
                              mode='bilinear', align_corners=False))
        mask = disp < p.test_disparity - 0.5
        disp[mask == 0] = 0
        masks.append(mask)
        disp = ((disp + 0.5) // 2**i).long()  # (N, 1, H, W)
        disp = one_hot(disp, p.test_disparity // 2**i, dim=1)
        disps.append(disp)

    losses = []
    for cost_vol, disp, mask in zip(cost_vols, disps, masks):
        losses.append(criterion(cost_vol, disp, mask))

    return [cost_vol.cpu().numpy() for cost_vol in cost_vols], [loss.cpu().numpy() for loss in losses], \
        [kernelL.cpu().numpy() if kernelL is not None else None for kernelL in kernelsL], \
        [kernelR.cpu().numpy() if kernelR is not None else None for kernelR in kernelsR]


pre_epoch = 0
pre_step = 0
min_loss = float('inf')
try:
    checkpoint = torch.load(p.SAVE_PATH + 'checkpoint.tar')
    pre_epoch = checkpoint['epoch']
    pre_step = checkpoint['step']
    min_loss = checkpoint['min_loss']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("optim")
    head.load_state_dict(checkpoint['model_state_dict'])
    print("head")
    print("restored successfully.. start from {} epoch {} step, min_loss: {}".format(
        pre_epoch, pre_step, min_loss))
except:
    print("error")

if p.mode == 'train':
    train_dataset = MyDataset(
        'train', transform=transform, normalize=True, crop_size=p.train_size)
    validate_dataset = MyDataset(
        'train', transform=transform, normalize=True, crop_size=p.train_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)
    validate_dataloader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=p.batch_size, shuffle=True, num_workers=1)

    # 필터는 냅두고 커널만 훈련함.
    """
    if p.train_guide:
        for param in head.parameters():
            param.requires_grad = False
    """

    for epoch in range(pre_epoch, 500):
        train_data_iter = iter(train_dataloader)

        for step in range(pre_step, len(train_dataloader)):
            data = next(train_data_iter)
            guideL = torch.tensor(np.expand_dims(cv2.Canny(np.uint8(np.transpose(np.squeeze(data['dispL'].numpy(
            ), axis=0), (1, 2, 0))), 5, 10), axis=(0, 1)) / 255, dtype=torch.float, device=p.device)
            guideR = torch.tensor(np.expand_dims(cv2.Canny(np.uint8(np.transpose(np.squeeze(data['dispR'].numpy(
            ), axis=0), (1, 2, 0))), 5, 10), axis=(0, 1)) / 255, dtype=torch.float, device=p.device)
            losses = train(data['imL_gray'], data['imR_gray'],
                           guideL, guideR, data['dispL'], guide=p.train_guide)
            if step % 20 == 0:
                print_losses("step: " + str(step), losses)

            if step % 500 == 0:
                validate_data_iter = iter(validate_dataloader)
                validate_steps = 100
                validate_steps = min(validate_steps, len(validate_dataloader))
                validate_losses = [0., 0., 0., 0., 0.]
                for val_step in range(validate_steps):
                    data = next(validate_data_iter)
                    guideL = torch.tensor(np.expand_dims(cv2.Canny(np.uint8(np.transpose(np.squeeze(data['dispL'].numpy(
                    ), axis=0), (1, 2, 0))), 5, 10), axis=(0, 1)) / 255, dtype=torch.float, device=p.device)
                    guideR = torch.tensor(np.expand_dims(cv2.Canny(np.uint8(np.transpose(np.squeeze(data['dispR'].numpy(
                    ), axis=0), (1, 2, 0))), 5, 10), axis=(0, 1)) / 255, dtype=torch.float, device=p.device)
                    losses = validate(data['imL_gray'], data['imR_gray'],
                                      guideL, guideR, data['dispL'], guide=p.train_guide)

                    for i in range(len(losses)):
                        validate_losses[i] += losses[i] / validate_steps

                print_losses("\nvalidate loss", losses, "\n")

                if min_loss >= sum(losses):
                    min_loss = sum(losses)
                    torch.save({'epoch': epoch,
                                'step': step + 1,
                                'min_loss': min_loss,
                                'model_state_dict': head.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               p.SAVE_PATH + 'checkpoint' + str(epoch) + 'ep' + str(step) + 'step' + str(sum(losses)) + 'loss.tar')

                torch.save({'epoch': epoch,
                            'step': step + 1,
                            'min_loss': min_loss,
                            'model_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           p.SAVE_PATH + 'checkpoint.tar')

        pre_step = 0

if p.mode == 'test':
    # test_dataset = Middlebury('train', transform=transform, normalize=True, resize=p.test_size)
    test_dataset = MyDataset('train', transform=transform,
                             normalize=True, resize=p.test_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=p.batch_size, shuffle=False, num_workers=1)

    test_data_iter = iter(test_dataloader)

    bad_h_0, bad_h_1, bad_h_2, bad_h_3, bad_h_4 = [], [], [], [], []
    bad_1_0, bad_1_1, bad_1_2, bad_1_3, bad_1_4 = [], [], [], [], []
    bad_2_0, bad_2_1, bad_2_2, bad_2_3, bad_2_4 = [], [], [], [], []
    bad_4_0, bad_4_1, bad_4_2, bad_4_3, bad_4_4 = [], [], [], [], []
    bad_8_0, bad_8_1, bad_8_2, bad_8_3, bad_8_4 = [], [], [], [], []
    avgerr_0, avgerr_1, avgerr_2, avgerr_3, avgerr_4 = [], [], [], [], []
    rms_0, rms_1, rms_2, rms_3, rms_4 = [], [], [], [], []

    plt.rc('xtick', labelsize=4)
    plt.rc('ytick', labelsize=4)
    plt.subplots_adjust(left=0.02, bottom=0, right=0.98,
                        top=1, wspace=0.08, hspace=0)

    for step in range(len(test_dataloader)):
        data = next(test_data_iter)
        guideL = torch.tensor(np.expand_dims(cv2.Canny(np.uint8(np.transpose(np.squeeze(data['dispL'].numpy(
        ), axis=0), (1, 2, 0))), 5, 10), axis=(0, 1)) / 255, dtype=torch.float, device=p.device)
        guideR = torch.tensor(np.expand_dims(cv2.Canny(np.uint8(np.transpose(np.squeeze(data['dispR'].numpy(
        ), axis=0), (1, 2, 0))), 5, 10), axis=(0, 1)) / 255, dtype=torch.float, device=p.device)
        cost_vols, losses, kernelsL, kernelsR = test(
            data['imL_gray'], data['imR_gray'], guideL, guideR, data['dispL'], guide=p.train_guide)

        batch = 0
        # scale_level = 4

        cost_vols = [np.transpose(cost_vol[batch], (1, 2, 0))
                     for cost_vol in cost_vols]  # (N, D, H, W) => (H, W, D)
        # (N, C=1, KH, KW, H, W) => (H, W, KH, KW, C=1)
        kernelsL = [np.transpose(kernelL[batch], (3, 4, 1, 2, 0))
                    if kernelL is not None else None for kernelL in kernelsL]
        kernelsR = [np.transpose(kernelR[batch], (3, 4, 1, 2, 0))
                    if kernelR is not None else None for kernelR in kernelsR]
        # (N, C, H, W) => (H, W, C)
        imageL = np.transpose(data['imL_raw'].numpy()[batch], (1, 2, 0))
        imageR = np.transpose(data['imR_raw'].numpy()[batch], (1, 2, 0))
        dispL = np.transpose(data['dispL_raw'].numpy()[batch], (1, 2, 0))
        preds = [np.argmax(cost_vols[i], -1) for i in range(5)]

        ax_height = 2
        ax_width = 3
        ax_num = 6

        for scale_level in range(5):

            axes = [plt.subplot(ax_height, ax_width, i + 1)
                    for i in range(ax_num)]

            def get_event_data(event):
                return event.inaxes, event.xdata, event.ydata, event.button

            def mouse_click_event(event):
                inaxes, x, y, button = get_event_data(event)

                if button == 3:  # 오른쪽 버튼
                    if inaxes in axes[0:2]:
                        print("x:", x, "y:", y)
                        if kernelsL[scale_level] is not None:
                            plot_kernel(axes[0], imageL, p.test_size,
                                        kernelsL[scale_level], (x, y))
                        if kernelsR[scale_level] is not None:
                            plot_kernel(axes[1], imageR, p.test_size,
                                        kernelsR[scale_level], (x, y))
                        plt.draw()
                    if inaxes in axes[2:3]:  # new
                        X = x
                        Y = y
                        x = int(x / imageL.shape[1] * p.test_size[1])
                        y = int(y / imageL.shape[0] * p.test_size[0])
                        print("x:", x, "y:", y)
                        if kernelsL[scale_level] is not None:
                            plot_kernel(axes[0], imageL, p.test_size,
                                        kernelsL[scale_level], (x, y))
                        if kernelsR[scale_level] is not None:
                            p_disp = dispL[int(Y+0.5), int(X+0.5)] / \
                                dispL.shape[1] * p.test_size[1]
                            print("disp:", p_disp)
                            plot_kernel(axes[1], imageR, p.test_size,
                                        kernelsR[scale_level], (x - p_disp, y))
                        plt.draw()
                    if inaxes in axes[3:5]:
                        x = int(x / imageL.shape[1] * p.test_size[1])
                        y = int(y / imageL.shape[0] * p.test_size[0])
                        print("x:", x, "y:", y)
                        if kernelsL[scale_level] is not None:
                            plot_kernel(axes[0], imageL, p.test_size,
                                        kernelsL[scale_level], (x, y))
                        if kernelsR[scale_level] is not None:
                            p_disp = preds[scale_level][y // 2**scale_level,
                                                        x // 2**scale_level] * 2**scale_level
                            print("disp:", p_disp)
                            plot_kernel(axes[1], imageR, p.test_size,
                                        kernelsR[scale_level], (x - p_disp, y))
                        plt.draw()
                        plot_costgraph(
                            cost_vols[scale_level], x // 2**scale_level, y // 2**scale_level, dispL)  # new

            plot_all(axes, imageL, imageR, dispL, cost_vols[scale_level], p.test_size, p.test_disparity,
                     kernelsL[scale_level], kernelsR[scale_level], (p.test_size[1] // 2, p.test_size[0] // 2))
            plt.connect('button_press_event', mouse_click_event)

            bad_h, bad_1, bad_2, bad_4, bad_8, avgerr, rms = plot_pred_error(
                None, dispL, cost_vols[scale_level], p.test_size, p.test_disparity)
            if scale_level == 0:
                bad_h_0.append(bad_h)
                bad_1_0.append(bad_1)
                bad_2_0.append(bad_2)
                bad_4_0.append(bad_4)
                bad_8_0.append(bad_8)
                avgerr_0.append(avgerr)
                rms_0.append(rms)
            if scale_level == 1:
                bad_h_1.append(bad_h)
                bad_1_1.append(bad_1)
                bad_2_1.append(bad_2)
                bad_4_1.append(bad_4)
                bad_8_1.append(bad_8)
                avgerr_1.append(avgerr)
                rms_1.append(rms)
            if scale_level == 2:
                bad_h_2.append(bad_h)
                bad_1_2.append(bad_1)
                bad_2_2.append(bad_2)
                bad_4_2.append(bad_4)
                bad_8_2.append(bad_8)
                avgerr_2.append(avgerr)
                rms_2.append(rms)
            if scale_level == 3:
                bad_h_3.append(bad_h)
                bad_1_3.append(bad_1)
                bad_2_3.append(bad_2)
                bad_4_3.append(bad_4)
                bad_8_3.append(bad_8)
                avgerr_3.append(avgerr)
                rms_3.append(rms)
            if scale_level == 4:
                bad_h_4.append(bad_h)
                bad_1_4.append(bad_1)
                bad_2_4.append(bad_2)
                bad_4_4.append(bad_4)
                bad_8_4.append(bad_8)
                avgerr_4.append(avgerr)
                rms_4.append(rms)

            # plt.tight_layout()
            plt.subplots_adjust(left=0.02, bottom=0,
                                right=0.98, top=1, wspace=0.08, hspace=0)
            # plt.savefig('/home/gu/workspace/Gu/EANet/EANet_0.1-5/result/base_ex2/step' + str(step) + 'scale' + str(scale_level) + '.png', dpi=300)
            plt.show()

            print("\n")

    weight = [1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 1, 1, 0.5, 1, 0.5]
    print("bad_h_0:", np.average(bad_h_0, 0, weight))
    print("bad_1_0:", np.average(bad_1_0, 0, weight))
    print("bad_2_0:", np.average(bad_2_0, 0, weight))
    print("bad_4_0:", np.average(bad_4_0, 0, weight))
    print("bad_8_0:", np.average(bad_8_0, 0, weight))
    print("avgerr_0:", np.average(avgerr_0, 0, weight))
    print("rms_0:", np.average(rms_0, 0, weight))
    print("------------------------------------------")
    print("bad_h_1:", np.average(bad_h_1, 0, weight))
    print("bad_1_1:", np.average(bad_1_1, 0, weight))
    print("bad_2_1:", np.average(bad_2_1, 0, weight))
    print("bad_4_1:", np.average(bad_4_1, 0, weight))
    print("bad_8_1:", np.average(bad_8_1, 0, weight))
    print("avgerr_1:", np.average(avgerr_1, 0, weight))
    print("rms_1:", np.average(rms_1, 0, weight))
    print("------------------------------------------")
    print("bad_h_2:", np.average(bad_h_2, 0, weight))
    print("bad_1_2:", np.average(bad_1_2, 0, weight))
    print("bad_2_2:", np.average(bad_2_2, 0, weight))
    print("bad_4_2:", np.average(bad_4_2, 0, weight))
    print("bad_8_2:", np.average(bad_8_2, 0, weight))
    print("avgerr_2:", np.average(avgerr_2, 0, weight))
    print("rms_2:", np.average(rms_2, 0, weight))
    print("------------------------------------------")
    print("bad_h_3:", np.average(bad_h_3, 0, weight))
    print("bad_1_3:", np.average(bad_1_3, 0, weight))
    print("bad_2_3:", np.average(bad_2_3, 0, weight))
    print("bad_4_3:", np.average(bad_4_3, 0, weight))
    print("bad_8_3:", np.average(bad_8_3, 0, weight))
    print("avgerr_3:", np.average(avgerr_3, 0, weight))
    print("rms_3:", np.average(rms_3, 0, weight))
    print("------------------------------------------")
    print("bad_h_4:", np.average(bad_h_4, 0, weight))
    print("bad_1_4:", np.average(bad_1_4, 0, weight))
    print("bad_2_4:", np.average(bad_2_4, 0, weight))
    print("bad_4_4:", np.average(bad_4_4, 0, weight))
    print("bad_8_4:", np.average(bad_8_4, 0, weight))
    print("avgerr_4:", np.average(avgerr_4, 0, weight))
    print("rms_4:", np.average(rms_4, 0, weight))
