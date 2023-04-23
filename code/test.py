from ACASX import ACASX
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier,export_text
from tqdm import tqdm
import os
import operator
from property import property
import time
time_stamp={
        "I>B":[],
        "A>I":[],
        "ovr":[]
    }
def read_data(path):
    acas_train = np.empty([384221,5],dtype=float)
    acas_train_labels = np.zeros(384221,dtype=int)
    num=0
    with open(path) as f:
        lines = f.readlines()
        print(len(lines), "examples")
        acas_train = np.empty([len(lines),5],dtype=float)
        acas_train_labels = np.zeros(len(lines),dtype=int)
        
        for l in range(len(lines)):
            k = [float(stringIn) for stringIn in lines[l].split(',')] #This is to remove the useless 1 at the start of each string. Not sure why that's there.
            #acas_train[l+num] = np.zeros(5,dtype=float) #we're asuming that everything is 2D for now. The 1 is just to keep numpy happy.
            if len(k) > 5:
              lab = int(k[5])
              #if ((lab == 0) or (lab == 2)):
              #  lab = 0
              #else:
              #  lab = 1
              acas_train_labels[l+num] = lab
            
            count = 0
            for i in range(0,5):
                #print(count)
                acas_train[l+num][i] = k[i]
                
                #print(k[i])
            
    return acas_train,acas_train_labels

def get_all_invariants(estimator):
    def is_leaf(node):
        return estimator.tree_.children_left[node] == estimator.tree_.children_right[node]

    def left_child(node):
        return estimator.tree_.children_left[node]

    def right_child(node):
        return estimator.tree_.children_right[node]
    
    def get_all_paths_rec(node):
        # Returns a list of triples corresponding to paths
        # in the decision tree. Each triple consists of
        # - neurons encountered along the path
        # - signature along the path
        # - prediction class at the leaf
        # - number of training samples that hit the path
        # The prediction class and number of training samples
        # are set to -1 when the leaf is "impure".
        feature = estimator.tree_.feature
        if is_leaf(node):
            values = estimator.tree_.value[node][0]
            if len(np.where(values != 0)[0]) == 1:
                cl = estimator.classes_[np.where(values != 0)[0][0]]
                nsamples = estimator.tree_.n_node_samples[node]
            else:
                # impure node
                cl = -1
                nsamples = -1
            return [[[], [], cl, nsamples]]
            # If it is not a leaf both left and right childs must exist
        paths = [[[feature[node]] + p[0], [0] + p[1], p[2], p[3]] for p in get_all_paths_rec(left_child(node))]
        paths += [[[feature[node]] + p[0], [1] + p[1], p[2], p[3]] for p in get_all_paths_rec(right_child(node))]
        return paths
    paths =  get_all_paths_rec(0)
    print("Obtained all paths")
    invariants = {}
    for p in tqdm(paths):
        neuron_ids, neuron_sig, cl, nsamples = p
        if cl not in invariants:
            invariants[cl] = []
        # cluster = get_suffix_cluster(neuron_ids, neuron_sig)
        invariants[cl].append([neuron_ids, neuron_sig, nsamples])
    for cl in invariants.keys():
        invariants[cl] = sorted(invariants[cl], key=operator.itemgetter(2), reverse=True)
    return invariants
def property_membership(pr,data):
    inputsProp = []
    for i in range(0,len(data)):
        inp = data[i]
        if ((pr == 6) and ((inp[0] < -0.12929) or (inp[0] > 0.700435))):#p6
            continue
        if ((pr == 10) and ((inp[0] < 0.268978) or (inp[0] > 0.679858))):#p10
            continue
        if ((pr == 9) and ((inp[0] < -0.29523) or (inp[0] > -0.21226))):#p9
            continue
        if ((pr == 5) and ((inp[0] < -0.32427) or (inp[0] > -0.32179))):#p5
            continue
        if ((pr == 8) and ((inp[0] < -0.32842) or (inp[0] > 0.679858))):#p8
            continue
        if ((pr == 7) and ((inp[0] < -0.32842) or (inp[0] > 0.679858))):#p7
            continue
        if (((pr == 2) or (pr == 1)) and ((inp[0] < 0.6) or (inp[0] > 0.679858))):#p2, p1
            continue
        #if ((inp[0] < 0.6) or (inp[0] > 0.679858) ):#p2a
        if ((pr == 4) and ((inp[0] < -0.30353) or (inp[0] > -0.29855))):#p4
            continue

        
        if ((pr == 6) and ((inp[1] < -0.5) or (inp[1] > -0.11141))):#p6
            continue
        if ((pr == 10) and ((inp[1] < 0.111408) or (inp[1] > 0.499999))):#p10
            continue
        if ((pr == 9) and ((inp[1] < -0.06366) or (inp[1] > -0.02228))):#p9
            continue
        if ((pr == 5) and ((inp[1] < 0.031831) or (inp[1] > 0.063662))):#p5
            continue
        if ((pr == 8) and ((inp[1] < -0.5) or (inp[1] > -0.375))):#p8
            continue
        if ((pr == 7) and ((inp[1] < -0.5) or (inp[1] > 0.499999))):#p7
            continue
        if (((pr == 2) or (pr == 1)) and ((inp[1] < -0.5) or (inp[1] > 0.5 ))):#p2, p1
            continue
        #if ((inp[0] < 0.6) or (inp[0] > 0.679858) ):#p2a
        if ((pr == 4) and ((inp[1] < -0.00955) or (inp[1] > 0.009549 ))):#p4
            continue


        if ((pr == 6) and ((inp[2] < -0.5) or (inp[2] > -0.4992))):#p6
            continue
        if ((pr == 10) and ((inp[2] < -0.5) or (inp[2] > -0.49841))):#p10
            continue
        if ((pr == 9) and ((inp[2] < -0.5) or (inp[2] > -0.49841))):#p9
            continue
        if ((pr == 5) and ((inp[2] < -0.5) or (inp[2] > -0.4992) )):#p5
            continue
        if ((pr == 8) and ((inp[2] < -0.01592) or (inp[2] > 0.015915))):#p8
            continue
        if ((pr == 7) and ((inp[2] < -0.5) or (inp[2] > 0.499999) )):#p7
            continue
        if (((pr == 2) or (pr == 1)) and ((inp[2] < -0.5) or (inp[2] > 0.5) )):#p2, p1
            continue
        #if ((inp[0] < 0.6) or (inp[0] > 0.679858) ):#p2a
        if ((pr == 4) and ((inp[2] < 0.493379))):#p4
            continue


        if ((pr == 6) and ((inp[3] < -0.5) or (inp[3] > 0.5))):#p6
            continue
        if ((pr == 10) and ((inp[3] < 0.227273) or (inp[3] > 0.5))):#p10
            continue
        if ((pr == 9) and ((inp[3] < -0.5) or (inp[3] > -0.45455))):#p9
            continue
        if ((pr == 5) and ((inp[3] < -0.5) or (inp[3] > -0.22727) )):#p5
            continue
        if ((pr == 8) and ((inp[3] < -0.045) or (inp[3] > 0.5))):#p8
            continue
        if ((pr == 7) and ((inp[3] < -0.5) or (inp[3] > 0.5))):#p7
            continue
        if (((pr == 2) or (pr == 1)) and ((inp[3] < 0.45) or (inp[3] > 0.5))):#p2, p1
            continue
        #if ((inp[0] < 0.6) or (inp[0] > 0.679858) ):#p2a
        if ((pr == 4) and (inp[3] < 0.3 )):#p4
            continue

        if ((pr == 6) and ((inp[4] < -0.5) or (inp[4] > 0.5))):#p6
            continue
        if ((pr == 10) and ((inp[4] < 0) or (inp[4] > 0.5) )):#p10
            continue
        if ((pr == 9) and ((inp[4] < -0.5) or (inp[4] > -0.375))):#p9
            continue
        if ((pr == 5) and ((inp[4] < -0.5) or (inp[4] > -0.16667))):#p5
            continue
        if ((pr == 8) and ((inp[4] < 0.0) or (inp[4] > 0.5))):#p8
            continue
        if ((pr == 7) and ((inp[4] < -0.5) or (inp[4] > 0.5) )):#p7
            continue
        if (((pr == 2) or (pr == 1)) and ((inp[4] < -0.5) or (inp[4] > -0.45))):#p2, p1
            continue
        #if ((inp[0] < 0.6) or (inp[0] > 0.679858) ):#p2a
        if ((pr == 4) and (inp[4] < 0.3 )):#p4
            continue

        inputsProp.append(i)

    return inputsProp
def property_chk(pr,label,all_invariants,prop_minIn,prop_maxIn,lay,suffixes,data,model):
    global time_stamp
    
    inputsProp = property_membership(pr,data)
    print("Property:" + str(pr), ",A => LABEL" + str(label))
    print('INPUTS WITHIN A:', len(inputsProp))
    
    df = []
    print(len(inputsProp))
    images = data
    imgsCom = []
    tot = 0
    notCovered = inputsProp
    notCov = []
    for cl, invs in all_invariants.items():
        
        if (cl != label):#if the end res is not satisfy the lable
            continue
        print(cl, len(invs))
    
        cnt = 0
        
        for invariant in invs:
            cls = get_suffix_cluster(invariant[0],invariant[1],suffixes)
            #get input that satisfy the given pattern
            lst3 = [value for value in inputsProp if value in cls]#get input satisfy the pattern and have feasible output
            withinAcnt = len(lst3)
            notCovered = list(set(notCovered) - set(lst3))            
            if (withinAcnt > 100):
                print("")
                print('INVARIANT > 100:' , invariant[0], invariant[1])
                print('SUPPORT > 100:' , invariant[2], ",", len(cls),', A SUPPORT:', withinAcnt)                
                minI = np.zeros(5)
                maxI = np.zeros(5)
                for ind in range(0,5):
                    minI[ind] = 1000
                    maxI[ind] = -1000                
                imgs = []
                print('COVERED:')
                for indx in range(0,len(lst3)):
                    index = lst3[indx]
                    img = data[index]
                    imgs.append(img)
                    #print(list(img))
                    for ind in range(0,5):
                        if (img[ind] < minI[ind]):
                            minI[ind] = img[ind]
                        if (img[ind] > maxI[ind]):
                            maxI[ind] = img[ind]
                print('INP MIN,MAX covered:')
                print(list(minI))
                print(list(maxI))
                if (len(notCovered) > 0):
                    for ind in range(0,5):
                        minI[ind] = 1000
                        maxI[ind] = -1000
                    print('NOT COVERED:')
                    for indx in range(0,len(notCovered)):
                        index = notCovered[indx]
                        img = data[index]
                        #print(list(img))
                    for ind in range(0,5):
                        if (img[ind] < minI[ind]):
                            minI[ind] = img[ind]
                        if (img[ind] > maxI[ind]):
                            maxI[ind] = img[ind] 
                    print('INP MIN,MAX not covered:')
                    print(list(minI))
                    print(list(maxI))


                # Get the min,max before the layer
                minIn =[]
                maxIn = []
                for ind in range(0,50):
                    minIn.append(1000)
                    maxIn.append(-1000)
                imgs=torch.tensor(imgs,dtype=torch.float32)
                layer_vals = model.get_layer(imgs,5).detach().numpy()
                print('MIN, MAX LAYER INPS:',len(layer_vals))
                for i in range(0,len(layer_vals)):
                    for dim in range(0,50):
                        if ( layer_vals[i][dim] < minIn[dim]):
                                        minIn[dim] = layer_vals[i][dim]
                        if ( layer_vals[i][dim] > maxIn[dim]):
                                        maxIn[dim] = layer_vals[i][dim]
        
                print('INP MIN,MAX PREV LAYER:')
                print(list(minIn))
                print(list(maxIn))
    
                
                sig = []
                for dim in range(0,50):
                    if (layer_vals[0][dim] == 0):
                        sig.append(0)
                    else:
                        sig.append(1)
                    

                for i in range(1,len(layer_vals)):
                    #print(layer_vals[i][0],layer_vals[i][47],layer_vals[i][48] )
                    for dim in range(0,50):
                        if (sig[dim] == -1):
                            continue
                        if ((layer_vals[i][dim] > 0) and (sig[dim] == 0)):
                            sig[dim] = -1
                            continue
                        if ((layer_vals[i][dim] == 0) and (sig[dim] == 1)):
                            sig[dim] = -1
                            continue

                print("COMMON SIGNATURE:")
                for dim in range(0,50):
                    if (sig[dim] != -1):
                        print("index:" + str(dim) + "=" + str(sig[dim]))
                
                neurons = invariant[0]
                signature = invariant[1]
                t=time.time()
                prov = False
                print(" CHECK I /\ covered_consts on short network => B") ## UPDATE notCov
                time_stamp["I>B"].append(time.time()-t)
                prov = invoke_marabou_chk(lay,neurons,signature,label,minIn,maxIn,sig) ## I => LABEL
    
                if (prov == True):
                    
                    notCovStrs = []
                    notCovStrs.append(notCov)
                    for i in range(0,len(neurons)):
                        strChk = "ws_"+ str(lay) + "_" + str(neurons[i])
                        if (strChk in notCovStrs):
                            #smthing not right in here
                            notCov.remove(notCovstr)
                    
                    print(" CHECK A  => I ") ## COLLECT NOT COVERED CONSTRAINTS
                    t=time.time()
                    notCov.append(invoke_marabou_chk_2(lay,neurons,signature,prop_minIn,prop_maxIn)) 
                    time_stamp["A>I"].append(time.time()-t)
                if (len(notCov) == 0):       
                    break

def invoke_marabou_chk(layer,neurons,signature,label,inp_min = [],inp_max = [], com_sig = [], notCov = []):
    propsig = True
    not_done = True
    comsig = False
    while( (not_done == True) and (comsig == False)):
  
        for lab_indx in range(0,5):
            if (lab_indx == label):
                continue

            not_done = False
            strInp = ""
            if (len(inp_min) > 0):
                for i in range(0,50):
                    strInp = strInp + "x"+ str(i) + " >= " + str(inp_min[i]) + "\n"
                    strInp = strInp + "x"+ str(i) + " <= " + str(inp_max[i]) + "\n"
                #print(strInp)

            strInternal = ""
            if (propsig == True):
                for i in range(0,len(neurons)):
                    strInternal = strInternal + "ws_"+ str(1) + "_" + str(neurons[i])
                    if (signature[i] == 0):
                        strInternal = strInternal + " <= 0.0" + "\n"
                    else:
                        strInternal = strInternal + " >= 0.0"  + "\n"
                propsig = False
            else:
                for dim in range(0,len(com_sig)):
                    if (com_sig[dim] == -1):
                        continue
                    strInternal = strInternal + "ws_"+ str(1) + "_" + str(dim)
                    if (com_sig[dim] == 0):
                        strInternal = strInternal + " <= 0.0" + "\n"
                    else:
                        strInternal = strInternal + " >= 0.0"  + "\n"
                comsig = True

            strOP = "+y"+ str(lab_indx) + " -y" + str(label) + " <= -0.001" + "\n"

            #Write to a property file
            file1 = open('property.txt',"w")
            file1.writelines(strInp) 
            #file1.writelines(strInternal) 
            file1.writelines(strOP) 
            file1.close() 

            file1 = open('property.txt',"r")  
            print("PROPERTY FILE IS ")
            print(file1.read())
            file1.close()

            #!./marabou_DnC_InternalNodes.elf ./ACASXU_run2a_1_1_batch_2000.nnet ./property.txt --dnc --num-workers=4 --summary-file=summary1.txt
            os.system("./marabou_DnC_InternalNodes.elf ./ACASXU_run2a_1_2_batch_fc5OP.nnet ./property.txt --summary-file=summary1.txt")
            print("SUMMARY:")
            f = open('summary1.txt', 'r')
            file_contents = f.read()
            print (file_contents)
            f.close()
            if (file_contents.find('UNSAT') == -1):
                not_done = True
                break
        
    if (not_done == False):
        print("Property proved!")
        return True
    else:
        print ("PROPERTY COULD NOT BE PROVED:")
        return False
  
def invoke_marabou_chk_2(layer,neurons,signature,inp_min,inp_max):
    strInp = ""
    for i in range(0,5):
          strInp = strInp + "x"+ str(i) + " >= " + str(inp_min[i]) + "\n"
          strInp = strInp + "x"+ str(i) + " <= " + str(inp_max[i]) + "\n"
    #print(strInp)

    strInternals = []
    for i in range(0,len(neurons)):
        strInternal = "ws_"+ str(layer) + "_" + str(neurons[i])
        if (signature[i] == 0):
           strInternal = strInternal + " >= 0.0" + "\n"
        else:
           strInternal = strInternal + " <= 0.0"  + "\n"
        strInternals.append(strInternal)

    notCov = []
    for strInternal in strInternals:
        #Write to a property file
        file1 = open('property.txt',"w")
        file1.writelines(strInp) 
        file1.writelines(strInternal) 
        file1.close() 

        file1 = open('property.txt',"r")  
        print("PROPERTY FILE IS ")
        print(file1.read())
        file1.close()

        os.system("./marabou_DnC_InternalNodes.elf ./ACASXU_run2a_1_1_batch_2000.nnet ./property.txt --summary-file=summary1.txt")
        print("SUMMARY:")
        f = open('summary1.txt', 'r')
        file_contents = f.read()
        print (file_contents)
        f.close()
        if (file_contents.find('UNSAT') == -1):
            notCov.append(strInternal)

    return notCov
def get_suffix_cluster(neuron_ids, neuron_sig,suffixes):
    # Get the cluster of inputs that such that all inputs in the cluster
    # have provided on/off signature for the provided neurons.
    #
    # The returned cluster is an array of indices (into mnist.train.images).
    return np.where((suffixes[:, neuron_ids] == neuron_sig).all(axis=1))[0]
def main():
    global time_stamp
    test=ACASX()
    test.set_weight("./ACASX_layer.txt")
    print(test)
    x_test,y_test=read_data('./clusterinACAS_0_shrt.csv')
    x_test=torch.tensor(x_test,dtype=torch.float32)
    y_test=torch.tensor(y_test,dtype=torch.float32)
    print(x_test.dtype)
    
    with torch.no_grad():
        out = test(x_test)
        print(out.shape)
        print(y_test.shape)
        predictions = torch.argmin(out, 1)
        print(predictions.shape)
    # Calculate accuracy
    #print(predictions[:100])
    #print(y_test[:100])

    acc=0
    for i in range(len(y_test)):
        if y_test[i]==predictions[i]:
            acc+=1
    LAYER=5
    accuracy = float(acc/len(y_test))
    test_suffixes=(test.get_layer(x_test,LAYER)>0.0).int()
    print(test_suffixes.shape)
    print(accuracy)
    tree_feature=test_suffixes.numpy()
    tree_pred=predictions.numpy()
    basic_estimator=DecisionTreeClassifier()
    basic_estimator.fit(tree_feature,tree_pred)
    print(tree_feature, tree_pred)
    print(basic_estimator.get_depth())
    print("layer:", LAYER)
    #",label:", LABEL)
    invariants = get_all_invariants(basic_estimator)
    #describe_invariants_all_labels(invariants,prev_lay,curr_lay)
    x_testnp=x_test.numpy()
    pr = 10
    label = 0
    pr_minIn = [0.268978,0.111408,-0.5, 0.227273,0.0]
    pr_maxIn = [0.679858,0.499999,-0.49841,0.5,0.5]

    print("CHECK PROPERTY:", str(pr) )
    print("REGION A:")
    print(pr_minIn)
    print(pr_maxIn)
    print("LABEL B:", str(label))

    t=time.time()
    property_chk(pr,label,invariants,pr_minIn,pr_maxIn,LAYER,tree_feature,x_testnp,test)
    time_stamp["ovr"].append(time.time()-t)
    print(time_stamp)
    print(time_stamp["A>I"]/len(time_stamp["A>I"]))

    print(time_stamp["I>B"]/len(time_stamp["I>B"]))
    print("time total:", time_stamp['ovr'])
    print("time on mrb I>B", sum(time_stamp["I>B"]/time_stamp['ovr']))

    print("time on mrb A>I", sum(time_stamp["A>I"]/time_stamp['ovr']))
if __name__ == "__main__":
    main()