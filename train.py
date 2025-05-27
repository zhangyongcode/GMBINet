# upload
import time

import torch
from torch.optim import Adam
from data_loader import *
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_ssim
import torch.nn as nn
from Inference import Inference
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, DeepSupervisionLoss, SoftclDiceLoss, DiceFocalLoss, SSIMLoss
import random
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
RANDOM_SEED = 42  # any random number


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(RANDOM_SEED)

# ------- define loss function -------
bce_loss = nn.BCELoss(reduction="mean")

ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
dice_loss = DiceLoss(include_background=False, )


# dice_loss = TverskyLoss(include_background=False, alpha=0.6, beta=0.4)
# dice_loss = SoftclDiceLoss()


def get_edge(image_tensor):
    # 定义 Sobel 边缘检测算子
    sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=torch.float32, requires_grad=False).cuda()
    sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=torch.float32, requires_grad=False).cuda()

    # 将卷积核调整为适合卷积操作的形状
    sobel_kernel_x = sobel_kernel_x.view(1, 1, 3, 3, )

    sobel_kernel_y = sobel_kernel_y.view(1, 1, 3, 3)

    # 应用卷积操作提取边缘
    edges_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1)
    edges_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1)

    # 计算梯度幅值
    edges = edges_x * edges_x + edges_y * edges_y
    # edges = torch.sqrt(edges_magnitude.clone().detach())

    edges[edges > 0] = 1
    # print(edges.size())
    return edges


def hybrid_loss(pred, target):
    bce_out = bce_loss(pred, target)
    # ssim_out = 1 - ssim_loss(pred, target)
    dice_out = dice_loss(pred, target)

    loss = bce_out  + dice_out #+ ssim_out

    return loss


def multi_loss_function(predictions, labels_v, weights):
    loss = 0
    for index in range(len(predictions)):
        loss = loss + hybrid_loss(predictions[index], labels_v) * weights[index]
    return loss


def main(model, train_image_dir, train_label_dir, val_image_dir, val_label_dir, log_dir, save_name, weights,
         lr=4e-3, batch_size=32, epoch_num=1083, device=torch.device("cuda:0"), resume=None):
    # 加载数据

    train_dataset = SalObjDataset(image_dir=train_image_dir, label_dir=train_label_dir,
                                  transform=transforms.Compose([
                                      Rescale(512),
                                      ToTensor(flag=0)]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    test_dataset = SalObjDataset(image_dir=val_image_dir, label_dir=val_label_dir,
                                 transform=transforms.Compose([Rescale(512), ToTensor(flag=0)]))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

    # 加载模型
    if resume is not None:
        dict = torch.load(resume, map_location=device)
        model.load_state_dict(dict)
        print('restore successfully')
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    ite_num = 0
    best_metric, best_metric_iter = 0, 0
    for epoch in range(0, epoch_num):
        model.train()
        epoch_loss, epoch_iter = 0, 0
        epoch_start = time.time()
        for i, data in enumerate(train_dataloader):
            ite_num, epoch_iter = ite_num + 1, epoch_iter + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            predictions = model(inputs)
            loss = multi_loss_function(predictions, labels, weights=weights)

            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

            if ite_num % 100 == 0:  # save model every 1000 iterations
                val_metric = Inference(model=model, device=device, dataloader=test_dataloader, save_dir=log_dir,
                                       target_size=(512, 512))
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_iter = ite_num
                    save_path = os.path.join(log_dir, str(save_name) + "_best_model_" + ".pt")
                    torch.save(model.state_dict(), save_path)

                print(
                    f"current iter: {ite_num},"
                    f" mean iou: {val_metric:.4f},"
                    f" best mean dice: {best_metric:.4f},"
                    f" at iter: {best_metric_iter},"
                )
        print(epoch, 'average loss %.4f' % (epoch_loss / epoch_iter), 'epoch times %.4f' % (time.time() - epoch_start))


if __name__ == '__main__':
    train_image_dir = r'F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\SurfUpload\Dataset\SD-saliency-900\train\image'
    train_label_dir = r'F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\SurfUpload\Dataset\SD-saliency-900\train\label'
    val_image_dir = r'F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\SurfUpload\Dataset\SD-saliency-900\val\image'
    val_label_dir = r'F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\SurfUpload\Dataset\SD-saliency-900\val\label'
    log_dir = 'F:\ZhangyongDoctor\PaperUpload-zhangyong7630\SurfDetection\SurfUpload\GMBINet-upload-githup\logs'

    save_name = "GMBINet"
    resume = None
    lr = 4e-3
    batch_size = 32
    epoch_num = 800
    device = torch.device("cuda:0")

    from model.GMBINet import GMBINet
    model = GMBINet(init_filters=16, blocks_down=(1, 3, 4, 6, 3), out_channel=1, )
    weights = [1, 1, 1, 1, 1]

    main(model, train_image_dir, train_label_dir, val_image_dir, val_label_dir, log_dir, save_name, weights,
         lr=lr, batch_size=batch_size, epoch_num=epoch_num, device=device, resume=resume)
