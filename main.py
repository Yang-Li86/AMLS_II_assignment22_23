from src.CFG import CFG
from src.utils import *
from src.Data import TrainDataset, TestDataset, get_transforms
from src.model import *
from src.train import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ======================================================================================================================
# Data preprocessing

# all available data
data = pd.read_csv('Datasets/train-5kfold_remove_noisy_by_0622_rad_13_08_ka_15_10_1.csv', index_col='index')
# Set the file path
test_csv_path = 'Output/test.csv'

# Randomly select 500 rows from the dataframe
rows = data.sample(n=round(0.1*data.shape[0]))

# Save the selected rows to a new CSV file
rows.to_csv(test_csv_path, index=True, header=True)

# Delete the selected rows from the original dataframe
data.drop(rows.index, inplace=True)

# get test data
test = pd.read_csv('Output/test.csv', index_col='index')
test = test.reset_index(drop=True)

pretrained_path = {'se_resnext50_32x4d': 'Datasets/se_resnext50_32x4d-a260b3a4.pth'}


if CFG.debug:
    folds = data.sample(n=20, random_state=CFG.seed).reset_index(drop=True).copy()
else:
    folds = data.copy()
    folds = folds.reset_index(drop=True)

train_labels = folds[CFG.target_col].values
kf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_index, val_index) in enumerate(kf.split(folds.values, train_labels)):
    folds.loc[val_index, 'fold'] = int(fold)
folds['fold'] = folds['fold'].astype(int)
folds.to_csv('Output/folds.csv', index=None)

# ======================================================================================================================
# Training

for fold in range(CFG.n_fold):
    train_fn(fold, folds, device)

# ======================================================================================================================
# Testing