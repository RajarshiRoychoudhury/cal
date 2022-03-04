from torch.utils.data import Dataset


class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DatasetMapperDiscriminative(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx


class DatasetMapper2(Dataset):
    def __init__(self, x, y):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

class DatasetMapperDiscriminative2(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], idx