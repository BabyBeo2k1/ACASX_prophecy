from ACASX import ACASX
import numpy as np
import torch
def read_data(path):
    x,y=[],[]
    with open(path,'r') as f:
        lines=f.readlines()
        for line in lines:
            data=line.split(',')
            x.append([float(d) for d in data[:5]])
            """t=[0,0,0,0,0]
            t[int(float(data[5]))]=1
            y.append(t)"""
            y.append(int(float(data[5])))
    return x,y
def main():
    test=ACASX()
    test.set_weight("./ACASX_layer.txt")
    print(test)
    x_test,y_test=read_data('./clusterinACAS_0_shrt.csv')
    x_test=torch.tensor(x_test)
    y_test=torch.tensor(y_test)
    with torch.no_grad():
        out = test(x_test)
        print(out.shape)
        print(y_test.shape)
        predictions = torch.argmin(out, 1)
        print(predictions.shape)
    # Calculate accuracy
    print(predictions[:100])
    print(y_test[:100])

    acc=0
    for i in range(len(y_test)):
        if y_test[i]==predictions[i]:
            acc+=1

    accuracy = float(acc/len(y_test))
    print(accuracy)
main()