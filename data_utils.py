import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,dirs,transform=None):
        super(MyDataset, self).__init__()
        self.dirs = dirs
        self.transform = transform

    def __getitem__(self, index):
        dir = self.dirs[index]
        img = Image.open(dir).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img,dir

    def __len__(self):
        return len(self.dirs)

def getFileList(path):

    files = []
    for f in os.listdir(path):
        if not f.endswith("~") or not f == "":
            files.append(os.path.join(path, f))

    return files