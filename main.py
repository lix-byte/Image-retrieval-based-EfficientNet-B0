from torchvision import datasets,transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss.labelloss import LabelSmoothing
import argparse
from torch import optim
from model.model import MyEfficientNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train_val():
    model.train()
    best_acc = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        train_loss = 0.0
        train_count = 0
        for i,(inputs,labels) in enumerate(tqdm(trainloader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            _,pred = torch.max(outputs.data,1)
            train_loss += loss.item() * inputs.size(0)
            train_count += torch.sum(labels.data==pred)
        train_loss = train_loss/len(trainset.targets)
        train_acc = train_count.double()/len(trainset.targets)

        print('Train loss : {:.4f}, Train acc : {:.4f}'.format(train_loss,train_acc))

        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            model.eval()
            for j,(inputs,labels) in enumerate(tqdm(valloader)):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                _,pred = torch.max(outputs.data,1)
                val_loss += loss.item() * inputs.size(0)
                val_count += torch.sum(labels.data == pred)
            val_loss = val_loss / len(valset.targets)
            val_acc = val_count.double() / len(valset.targets)
            if best_acc < val_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), './results/best_model.pth')
            print('val loss : {:.4f} , val acc : {:.4f}'.format(val_loss, val_acc))



if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Classification for Retrieval')
    parser.add_argument('--learning_rate',default='0.01',type=float)
    parser.add_argument('--epochs',default='20',type=int)
    parser.add_argument('--smoothing',default='0.1',type=float)
    parser.add_argument('--batch_size',default='64',type=int)
    parser.add_argument('--trainpath',default='./Animals_classifier/train',type=str)
    parser.add_argument('--valpath',default='./Animals_classifier/val',type=str)
    opt = parser.parse_args()

    lr, epochs, smoothing, batch_size, trainpath, valpath = opt.learning_rate, opt.epochs, opt.smoothing, opt.batch_size, opt.trainpath, opt.valpath

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    )

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    )


    trainset = datasets.ImageFolder(trainpath,transform=train_transforms)
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2)

    valset = datasets.ImageFolder(valpath,transform=val_transforms)
    valloader = DataLoader(valset,batch_size=batch_size,shuffle=False,num_workers=2)

    model = MyEfficientNet()
    model = model.cuda()
    criterion = LabelSmoothing(smoothing=smoothing)
    optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False)

    train_val()
