from PIL import Image
import torchvision.transforms as Transforms
from loadfile import *
from torch.utils.data import DataLoader,random_split,Dataset

def preprocessing(image):

    Transforms_img = Transforms.Compose([
    Transforms.ToTensor(),
    Transforms.Resize((128,128)),
    # Transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    img_transform = Transforms_img(image)
    return img_transform


def encode_label(path_data):
    labels = [path.split('/')[-2] for path in path_data]
    unique_labels = set(labels)
    index_label = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [index_label[label] for label in labels]
    return numeric_labels


Transforms_img = Transforms.Compose([
    Transforms.ToTensor(),
    Transforms.Resize((128,128)),
    Transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


class Custom_data(Dataset):
  def __init__(self,path_datas,encode,Transforms_img):
    super().__init__()
    self.path_datas = path_datas
    self.encode = encode
    self.Transforms_img = Transforms_img
  def __len__(self):
    return len(self.path_datas)
  def __getitem__(self, index):
    img = self.Transforms_img(Image.open(self.path_datas[index]).convert('RGB'))
    label = torch.tensor(self.encode[index]).float()
    return img,label
  
def get_data():
  print(path_data_train[-1])
  label_train = encode_label(path_data_train)
  data_train = Custom_data(path_data_train,label_train,Transforms_img)
  print(data_train[-1][1])
  batch =64
  train_set,val_set = random_split(data_train,[0.8,0.2])
  train_loader = DataLoader(train_set,batch,shuffle=True)
  val_loader = DataLoader(val_set,batch,shuffle=True)
  return train_loader,val_loader
