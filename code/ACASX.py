import torch
import torch.nn as nn
import torch.nn.functional as F
class ACASX(nn.Module):
    
    def __init__(self):
        super(ACASX,self).__init__()
        self.fc0 =nn.Linear(5,50)
        self.fc1 =nn.Linear(50,50)
        self.fc2 =nn.Linear(50,50)
        self.fc3 =nn.Linear(50,50)
        self.fc4 =nn.Linear(50,50)
        self.fc5 =nn.Linear(50,50)
        self.output_layer = nn.Linear(50,5)
    def forward(self,input):
        x=F.relu(self.fc0(input))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        out =(self.output_layer(x))
        print(out.detach().mean().numpy().tolist())
        return out
    def get_layer(self,input,layer):
        self.eval()
        if layer>=1:
            x=F.relu(self.fc0(input))
        if layer>=2:
            x=F.relu(self.fc1(x))
        if layer>=3:    
            x=F.relu(self.fc2(x))
        if layer>=4:
            x=F.relu(self.fc3(x))
        if layer>=5:
            x=F.relu(self.fc4(x))
        if layer>=6:
            x=F.relu(self.fc5(x))
            
        return x
    def set_weight(self,path):
        with open (path) as f:
            lines = f.readlines()
            self.parameters
            weight=[]
            for indx in range(0,len(lines)):
                vals = lines[indx].split(',')
                #print vals
                weight.append(float(vals[4]))
            weight_tensor=torch.tensor(weight)
            count = 0
            params = self.parameters()
            for param in params:
                size = param.numel()
                param.data.copy_(weight_tensor[count:count + size].view_as(param))
                count += size
    