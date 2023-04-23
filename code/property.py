import numpy as np
class property:
    bound_ips=[]#constrain for input
    bound_ops=[]#constrain for output 
    input=[]#input membership
    output=[]#output membership
    pr_membership=[]
    def set_data(self,data):
        data=np.array(data)
        (x,y)=data
        x=np.array(x)
        y=np.array(y
                   )
        self.input=x.reshape((x.shape[0],-1))
        self.output=y.reshape((y.shape[0],-1))
    def __init__(self,bound_file) -> None:
        
        with open(bound_file,"r")as f:
            lines=f.readlines()
            for i in range(len(lines)):
                
                c=lines[i].split()
                
                if len(c)!=4:
                    continue
                if c[0]=="(assert":
                    if c[2][0]=="Y":
                        if c[1]=="(<=":
                            self.bound_x[int(c[2][-1])][1]=float(c[3][:-2])
                        if c[1]=="(>=":
                            self.bound_x[int(c[2][-1])][0]=float(c[3][:-2])
                    if c[2][0]=="X":
                        if c[1]=="(<=":
                            self.bound_x[int(c[2][-1])][1]=float(c[3][:-2])
                        if c[1]=="(>=":
                            self.bound_x[int(c[2][-1])][0]=float(c[3][:-2])
        
    def membership(self):
        # return the index of data that satisfy the constrain of the property
        # data shape is (inputs, outputs)
        # bound shape is (2, shape of single data)
        
        for i in range(len(self.input)):
            if (self.input[i]-self.bound_ips[i][0]).all()>0 and (self.input[i]-self.bound_ips[i][1]).all()<0 and (self.output[i]-self.bound_ops[i][0]).all()>0 and (self.output[i]-self.bound_ops[i][1]).all()<0:
                self.pr_membership.append(i)
        