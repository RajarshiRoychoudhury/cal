# from torch.utils.data import Dataset


# class DatasetMapper(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]


# class DatasetMapperDiscriminative(Dataset):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx], idx


# class DatasetMapper2(Dataset):
#     def __init__(self, x):
#         self.x = x

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx]

# class DatasetMapperDiscriminative2(Dataset):
#     def __init__(self, x):
#         self.x = x

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return self.x[idx], idx


from torch.utils.data import Dataset
import torch


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
    def __init__(self, x):
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        encodings = self.text[idx]
        #print(encodings["input_ids"].shape)
        item = {"input_ids": torch.tensor(encodings["input_ids"].squeeze(0))}
        item["attention_mask"] = torch.tensor(encodings["attention_mask"])
        item['labels'] = torch.tensor(self.labels[idx])
        return item["input_ids"], item["attention_mask"], item["labels"]

    def __len__(self):
        return len(self.labels)

class Dataset2(torch.utils.data.Dataset):
    def __init__(self, text):
        self.text = text

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        encodings = self.text[idx]
        #print(encodings["input_ids"].shape)
        item = {"input_ids": torch.tensor(encodings["input_ids"].squeeze(0))}
        item["attention_mask"] = torch.tensor(encodings["attention_mask"])
        return item["input_ids"], item["attention_mask"]

    def __len__(self):
        return len(self.text)

class DatasetDiscriminativeBERT(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encodings = self.text[idx]
        print(encodings)
        item = {"input_ids": torch.tensor(encodings["input_ids"].squeeze(0))}
        item["attention_mask"] = torch.tensor(encodings["attention_mask"])
        item['labels'] = torch.tensor(self.labels[idx])
        return (item["input_ids"], item["attention_mask"], item["labels"], idx)


class DatasetDiscriminativeBERT2(Dataset):
    def __init__(self, text):
        self.text = text
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encodings = self.text[idx]
        #print(encodings["input_ids"].shape)
        item = {"input_ids": torch.tensor(encodings["input_ids"].squeeze(0))}
        item["attention_mask"] = torch.tensor(encodings["attention_mask"])
        return (item["input_ids"], item["attention_mask"], idx)