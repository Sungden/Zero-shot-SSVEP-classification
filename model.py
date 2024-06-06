import torch
import torch.nn as nn
import numpy as np
from config import config
import matplotlib.pyplot as plt
from FBCCA import fbcca
from function import Ref_total


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(41)

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=(1,1))
        self.conv1_1 = nn.Conv2d(config.num_class, 1, kernel_size=(1,1))
        self.conv2 = nn.Conv2d(1, 120, kernel_size=(config.C, 1), bias=True)
        self.conv2_1 = nn.Conv2d(1, 120, kernel_size=(2*config.Nh, 1), bias=True)
        self.drop1 = nn.Dropout2d(0.1)
        self.conv3 = nn.Conv2d(120,120, kernel_size=(1, 2), stride=(1, 2),bias=True)
        self.conv3_1 = nn.Conv2d(120,120, kernel_size=(1, 2), stride=(1, 2),bias=True)
        self.conv3_2 = nn.Conv2d(120,120, kernel_size=(1, 2), stride=(1, 2),bias=True)
        self.drop2 = nn.Dropout2d(0.1)
        self.relu  = nn.ReLU()
        self.conv4 = nn.Conv2d(120, 120, kernel_size=(1, 10), padding='same', bias=True)
        self.conv4_1 = nn.Conv2d(120, 120, kernel_size=(1, 10), padding='same', bias=True)
        self.conv4_2 = nn.Conv2d(120, 120, kernel_size=(1, 10), padding='same', bias=True)
        self.drop3 = nn.Dropout2d(config.dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(120*(config.T//2), config.num_class)
        self.fc_1 = nn.Linear(120*(config.T//2), config.num_class)
        self.fc_2 = nn.Linear(120*(config.T//2), config.num_class)


    def forward(self, x,x_cca):    #torch.Size([batch_size, channel, samples]) 
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = self.drop1(x2)
        x3 = self.conv3(x2)
        x3 = self.drop2(x3)
        x3_=x3
        x3 = self.relu(x3)
        x4 = self.conv4(x3)
        x = self.drop3(x4)
        x = self.flatten(x)  # Flatten
        x = self.fc(x)

        # # x_cca = torch.unsqueeze(x_cca,dim=1)
        x_cca = self.conv1_1(x_cca)
        x_cca = self.conv2_1(x_cca)
        x_cca = self.drop1(x_cca)
        x_merge = self.conv3_2(x_cca+x2)

        x_cca = self.conv3_1(x_cca)
        x_cca = self.drop2(x_cca)
        x_merge = self.conv4_2(x_cca+x3_+x_merge)
        x_merge = self.drop3(x_merge)
        x_merge = self.flatten(x_merge)  # Flatten
        x_merge = self.fc_2(x_merge)


        x_cca = self.relu(x_cca)
        x_cca = self.conv4_1(x_cca)
        x_cca = self.drop3(x_cca)
        x_cca = self.flatten(x_cca)  # Flatten
        x_cca = self.fc_1(x_cca)
        return torch.squeeze(x)#+torch.squeeze(x_cca)+torch.squeeze(x_merge)

class universal_model:
    def __init__(self):
        self.batch_size=config.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.siamese = Siamese().to(device)
        self.optimizer1 = torch.optim.Adam(self.siamese.parameters(), lr=0.0002)

    def train(self,x,x_cca,y,label,n_epochs=100):  #size should be (block, channel, samples) (block,)
        x = torch.from_numpy(x)
        x_cca = torch.from_numpy(x_cca)
        y = torch.from_numpy(y)
        label=torch.from_numpy(label)

        dataset = torch.utils.data.TensorDataset(x, x_cca,y,label)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        optimizer =self.optimizer1
        model=self.siamese

        for epoch in range(n_epochs):
                model.train()
                loss=0
                accuracy=0

                for i, data in enumerate(dataloader, 0):
                    x, x_cca,y,labels = data
                    x = x.type(self.Tensor)
                    x_cca = x_cca.type(self.Tensor)
                    y = y.type(self.Tensor)

                    labels = labels.type(self.Tensor)
                    optimizer.zero_grad()
                    pred_x = model(x,x_cca) #torch.Size([batch_size, channels, samples])
                    loss_1 = self.criterion(pred_x,labels.long())

                    loss_total=loss_1
                    loss_total.backward()
                    optimizer.step()

                    accuracy += ((pred_x).argmax(dim=1) == labels).sum().cpu().item()
                    loss += loss_total.cpu().item()

                train_loss = loss / (i + 1)
                if (epoch+1)%50==0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [Accuracy: %f] [Loss: %f]"
                        % (epoch, config.n_epochs, i, len(dataloader), accuracy/len(label),train_loss)
                    ) 
        return model


    def test(self,model,testdata,testdata_cca): #size should be (channel, samples) 
        model.eval()
        testdata = torch.FloatTensor(testdata).to(device).type(self.Tensor)
        testdata_cca = torch.FloatTensor(testdata_cca).to(device).type(self.Tensor)
        pred_x= model(testdata,testdata_cca)
        prediction=pred_x.data.cpu().numpy()
        return prediction
