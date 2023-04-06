from sklearn.metrics import cohen_kappa_score
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from src.Data import TrainDataset, TestDataset, get_transforms
from src.model import CustomSEResNeXt
from src.CFG import CFG
from collections import Counter





def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


def train_fn(fold, folds, device):
    print(f"### fold: {fold} ###")

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_dataset = TrainDataset(folds.loc[trn_idx].reset_index(drop=True),
                                 folds.loc[trn_idx].reset_index(drop=True)[CFG.target_col],
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
                                 folds.loc[val_idx].reset_index(drop=True)[CFG.target_col],
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size)

    model = CustomSEResNeXt(model_name='se_resnext50_32x4d')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_score = -100
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        model.train()
        avg_loss = 0.

        optimizer.zero_grad()
        tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (images, labels) in tk0:
            images = images.to(device)
            labels = labels.to(device)

            y_preds = model(images)
            loss = criterion(y_preds, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item() / len(train_loader)

        model.eval()
        avg_val_loss = 0.
        preds = []
        valid_labels = []
        tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

        for i, (images, labels) in tk1:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                y_preds = model(images)

            preds.append(y_preds.to('cpu').numpy().argmax(1))
            valid_labels.append(labels.to('cpu').numpy())

            loss = criterion(y_preds, labels)
            avg_val_loss += loss.item() / len(valid_loader)

        scheduler.step(avg_val_loss)

        preds = np.concatenate(preds)
        valid_labels = np.concatenate(valid_labels)

        LOGGER.debug(f'Counter preds: {Counter(preds)}')
        score = quadratic_weighted_kappa(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.debug(
            f'  Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.debug(f'  Epoch {epoch + 1} - QWK: {score}')

        if score > best_score:
            best_score = score
            LOGGER.debug(f'  Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save(model.state_dict(), f'fold{fold}_se_resnext50.pth')