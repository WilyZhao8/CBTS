from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


class INaturalist(Dataset):
    def __init__(self, root, txt, transform=None, train=True):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 8142
        self.train = train
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            if self.train:
                h,w=sample.size
                ch = 0.3 * h
                cw = 0.3 * w
                sample1 = self.transform[0](sample)

                sample2 = self.transform[3](sample)
                sample3 = self.transform[3](sample)
                sample4 = self.transform[3](sample)
                sample5 = self.transform[3](sample)

                sample6 = self.transform[3](sample)
                sample7 = self.transform[3](sample)
                sample8 = self.transform[3](sample)
                sample9 = self.transform[3](sample)

                #sample4 = self.transform[2](sample)
                return [sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8, sample9], label  # , index
            else:
                return self.transform(sample), label