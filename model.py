import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from preprocessing import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        self.feature = nn.Sequential(
           nn.Conv2d(3,32,3,padding='same'), #b,6,128,128
           nn.ReLU(),
           nn.MaxPool2d(2), #b,6,64,64
           nn.BatchNorm2d(32),

           nn.Conv2d(32,64,3,padding='same'),#b,16,64,64
           nn.ReLU(),
           nn.MaxPool2d(2), #b,16,32,32
           nn.BatchNorm2d(64),

           nn.Conv2d(64,128,3,padding='same'),#b,16,64,64
           nn.ReLU(),
           nn.MaxPool2d(2), #b,16,32,32
           nn.BatchNorm2d(128)

        )

        self.fc = nn.Sequential(
            nn.Flatten(), #b,16*32*32
            nn.Linear(128*16*16,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,output),

        )
    def forward(self,x):
        feature = self.feature(x)
        return self.fc(feature)


model = Model(3,2).to(device)

from torch.optim import Adam
optimizer = Adam(model.parameters(), lr = 5e-5)



class crossentroy(nn.Module):
    def __init__(self,alpha):
        super().__init__()
    def forward(self,y_pred, y):
        b , num_class = y_pred.shape
        one_hot = torch.nn.functional.one_hot(y.long(),num_classes = num_class)
        softmax = torch.exp(y_pred)/(torch.sum(torch.exp(y_pred),-1)).view(-1,1)
        loss = - torch.log(torch.sum(softmax*one_hot,-1))
        return torch.mean(loss)
    
criterion = nn.CrossEntropyLoss()


class training:
    def __init__(self,**krawg):
        self.model = Model(3,2)
    def predict(self,image,device):
        with torch.no_grad():
            self.model.eval()
            image = image.to(device).unsqueeze(0)
            pred = self.model(image)
            print(pred)
            rs = torch.argmax(pred, -1).detach().clone().cpu().tolist()
        return rs
    
    def eval(self,val_loader, model,device):
        with torch.no_grad():
            model.eval()
            pred = []
            label = []
            for img_val,label_val in val_loader:
                img_val = img_val.to(device)
                label_val = label_val.to(device)
                pred_val = model(img_val)
                print(pred_val.shape)
                pred += torch.argmax(pred_val, -1).detach().clone().cpu().tolist()
                label +=label_val.detach().clone().cpu().tolist()
            return accuracy_score(pred,label)
        
    def train(self,epoch,train_loader,val_loader,device,criterion,optimizer):
        
        for e in tqdm(range(epoch)):
            self.model.train()
            total_loss = 0
            for img,label in train_loader:
                img = img.to(device)
                label = label.long().to(device)
                pred = self.model(img)
                loss = criterion(pred,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
            print(f'epoch: {e+1}---------total_loss: {total_loss/len(train_loader)}-----------  {self.eval(val_loader,self.model,device)}')
            

    def running(self,train_loader,val_loader,device,criterion,optimizer):
        self.train(2,train_loader,val_loader,device,criterion,optimizer)

    
