import pandas as pd
import torch


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        csv = pd.read_csv(file)
        x = csv.drop("label", axis=1).to_numpy()
        y = csv["label"].to_numpy()

        self.X_train = torch.tensor(x / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train )

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]