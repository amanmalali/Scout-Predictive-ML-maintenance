import torch.nn as nn
import torch

class basic_classifier(nn.Module):
    def __init__(self,input_size,output_size):
        super(basic_classifier,self).__init__()
        self.input_size=input_size
        self.lin1=nn.Linear(input_size,32)
        self.dropout1 = nn.Dropout(0.2)
        self.lin2=nn.Linear(32,64)
        self.dropout2 = nn.Dropout(0.2)
        self.lin3=nn.Linear(64,64)
        self.lin4=nn.Linear(64,output_size)

    def forward(self,x):
        x=torch.relu(self.lin1(x))
        x=self.dropout1(x)
        x=torch.relu(self.lin2(x))
        x=self.dropout2(x)
        x=torch.relu(self.lin3(x))
        conf=self.lin4(x)
        pred=torch.sigmoid(conf)
        return pred
    

        