#
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from data_loader import *
from torchvision import transforms


def IoU(prediction, label, smooth=0.0000001):
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    out = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return out


def Inference(model, dataloader, device=torch.device("cuda:0"), save_dir=None, target_size=(200, 200),
              save_flag=True):
    model.to(device)
    model.eval()

    IoU_list = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for index, data in enumerate(dataloader):
            image, label, name = data['image'], data['label'], data['name']
            image = image.type(torch.FloatTensor)
            image = image.to(device)
            prediction_list = model(image)  # 进行预测
            main_prediction = prediction_list[0]

            # 将结果从tensor 转为CPU上
            main_prediction, label = main_prediction.detach().cpu().numpy(), label.detach().cpu().numpy()
            main_prediction[main_prediction >= 0.5] = 1
            main_prediction[main_prediction < 0.5] = 0

            for i in range(main_prediction.shape[0]):
                pred = main_prediction[i, 0, :, :]
                if save_flag:
                    save_path = os.path.join(save_dir, name[i] + ".png")
                    cv2.imwrite(save_path, np.array(pred * 255, dtype='uint8'))
                    label_path = os.path.join(label_dir, name[i] + ".png")
                    label_data = cv2.imread(label_path)
                    shape = label_data.shape
                    img = cv2.imread(save_path)
                    img = cv2.resize(src=img, dsize=(shape[1], shape[0]))
                    cv2.imwrite(save_path, img)

                mask = label[i, 0, :, :]
                IoU_score = IoU(prediction=pred, label=mask)
                IoU_list.append(IoU_score)
    print(np.mean(IoU_list))
    return np.mean(IoU_list)


if __name__ == '__main__':
    image_dir = ''  # './dataset/NRSD_MN/test/image/'
    label_dir = ''  # './dataset/NRSD_MN/test/label/'
    prediction_dir = ''  #
    model_dir = ''  # "\logs\GBMINet-NRSD.pt"

    from model.GMBINet import GMBINet

    model = GMBINet(init_filters=16, blocks_down=(1, 3, 4, 6, 3), out_channel=1)

    model.load_state_dict(torch.load(model_dir))

    test_dataset = SalObjDataset(image_dir=image_dir, label_dir=label_dir,
                                 transform=transforms.Compose([Rescale(512), ToTensor(flag=0)]),
                                 image_ext='.jpg', label_ext='.png')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    Inference(model, test_dataloader, device=torch.device("cuda:0"), save_dir=prediction_dir,
              save_flag=True)
