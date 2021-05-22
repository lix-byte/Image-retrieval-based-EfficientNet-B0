import torch
from PIL import Image
from tqdm import tqdm

from data_utils import getFileList, MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from model.model import MyEfficientNet

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


model = MyEfficientNet()
model.load_state_dict(torch.load('./results/best_model.pth'))
model.classifier = nn.Sequential()

model = model.cuda()
model = model.eval()
def extract_features():
    features = torch.FloatTensor() #create an empty tensor,store features
    dir_list = []
    for img, dir in tqdm(dataloader):
        img = img.cuda()
        outputs = model(img)
        ff = outputs.data.cpu()
        features = torch.cat((features, ff), 0)
        dir_list += list(dir)
    return features, dir_list

def single_picture(query_path):
    img = Image.open(query_path)
    img = transform(img)
    img = img.cuda()
    img = img.unsqueeze(0)
    outputs = model(img)
    outputs = outputs.data.cpu()
    return outputs


if __name__=='__main__':

    path = r'./Animals'
    pathlist = getFileList(path)
    dataset = MyDataset(dirs=pathlist,transform=transform)
    dataloader = DataLoader(dataset,batch_size=8,shuffle=False,num_workers=2)
    features, dir_list = extract_features()
    data_base = {}
    data_base['img'] = dir_list
    data_base['features'] = features
    torch.save(data_base,'./database/database.pth')




