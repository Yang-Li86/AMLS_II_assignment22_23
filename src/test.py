import numpy as np
import torch
import os
from src.Data import *
from src.model import *
from torch.utils.data import DataLoader


def inference(model, test_loader, device):
    model.to(device)

    probs = []

    for i, images in enumerate(test_loader):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)

        probs.append(y_preds.to('cpu').numpy())

    probs = np.concatenate(probs)

    return probs

def submit(sample, dir_name='train_images'):
    if os.path.exists(f'../input/prostate-cancer-grade-assessment/{dir_name}'):
        print('run inference')
        test_dataset = TestDataset(sample, dir_name, transform=get_transforms(data='valid'))
        test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
        probs = []
        for fold in range(CFG.n_fold):
            model = CustomSEResNeXt(model_name='se_resnext50_32x4d')
            weights_path = f'fold{fold}_se_resnext50.pth'
            model.load_state_dict(torch.load(weights_path, map_location=device))
            _probs = inference(model, test_loader, device)
            probs.append(_probs)
        probs = np.mean(probs, axis=0)
        preds = probs.argmax(1)
        sample['isup_grade'] = preds
    return sample