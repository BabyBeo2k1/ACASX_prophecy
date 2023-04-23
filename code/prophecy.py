import torch
class Prophecy:
    def __init__(self) -> None:
        pass
    def get_pred(self,inps,layer=-1,net):
        out=net(inps)
        if layer==-1:
            return model.output_layer.output 