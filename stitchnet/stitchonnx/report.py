from .utils import accuracy_score_net, accuracy_score_net_plants, get_macs_params
from .viz import draw_stitchNet
import os
import traceback

class Report:
    def __init__(self, bs=128, fname=f'./_results/result_val.txt', mode='a'):
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        self.fname = fname
        self.mode = mode
        self.bs = bs
    def evaluate(self, nets, net, netname, score, dataset):
        dot, gname = draw_stitchNet(nets, net, name=netname)
        accuracy = accuracy_score_net(net, dataset, bs=self.bs)
        print('accuracy', accuracy)
        macs, params = get_macs_params(net[0])
        network = net.get_id()
        self.f.write(f'{score},{accuracy},{macs},{params},{gname},"{network}"\n')
        self.f.flush()
        
    def close(self):
        self.f.close()
    
    def __enter__(self):
        self.f = open(self.fname, self.mode)
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        self.close()
        return True
        
class ReportPlants:
    def __init__(self, bs=128, fname=f'./_results/result_val.txt', mode='a'):
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        self.fname = fname
        self.mode = mode
        self.bs = bs
    def evaluate(self, nets, net, netname, score, dataset, train_dataset):
        dot, gname = draw_stitchNet(nets, net, name=netname)
        accuracy = accuracy_score_net_plants(net, dataset, train_dataset, bs=self.bs)
        print('accuracy', accuracy)
        macs, params = get_macs_params(net[0])
        network = net.get_id()
        self.f.write(f'{score},{accuracy},{macs},{params},{gname},"{network}"\n')
        self.f.flush()
        
    def close(self):
        self.f.close()
    
    def __enter__(self):
        self.f = open(self.fname, self.mode)
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        self.close()
        return True
        

from evaluate import evaluator
from transformers import AutoProcessor
import onnxruntime as ort
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from .utils import change_input_dim, PROVIDERS, load_dl
from torch.utils.data import TensorDataset

class NetEvalPipeline:
    def __init__(self, train_dataset, net, label="label"):
        self.label = label
        self.net = net
        fragmentC = self.net[0]
        change_input_dim(fragmentC.fragment)
        self.ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=['CUDAExecutionProvider'])
        self.input_name = fragmentC.fragment.graph.input[0].name
        
        # self.p = AutoProcessor.from_pretrained('microsoft/resnet-50')
        
        nnX = []
        nnY = []
        ptdset_train = TensorDataset(train_dataset['pixel_values'], 
                             train_dataset[self.label])
        dl = load_dl(ptdset_train, batch_size=32, shuffle=False, num_workers=0)
        for dataItem in tqdm(dl):
        # for dataItem in tqdm(train_dataset):
            try:
                X = dataItem[0].squeeze(1).numpy()
                t = dataItem[1].unsqueeze(1).numpy()
                # print(t.shape)
                # pilImage,t = dataItem['image'],dataItem['label']
                # out = self.p(pilImage)
                # X = out['pixel_values']
                # X,t = dataItem['pixel_values'],dataItem['label']
                # X = X.numpy()
                inputs = {}
                inputs[self.input_name] = X
                outputs = self.ort_sess1.run(None, inputs)
                y = outputs[0]
                nnX.append(y)
                nnY.append(t)
            except Exception as e:
                print('ERROR TRAIN', e)
                traceback.print_exc()
        
        nnX = np.vstack(nnX)
        nnY = np.vstack(nnY).squeeze()
        print(nnX.shape, nnY.shape)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(nnX, nnY)
        
        self.knn = knn
        self.task = "image-classification"

    def __call__(self, pixel_values, **kwargs):
        result = []
        for pixel_value in tqdm(load_dl(pixel_values, batch_size=32, shuffle=False, num_workers=0)):
        # for pixel_value in tqdm(pixel_values):
            try:
                X = pixel_value.squeeze(1).numpy()
                print(X.shape)
                inputs = {}
                inputs[self.input_name] = X
                outputs = self.ort_sess1.run(None, inputs)
                y = outputs[0]
                y = np.array(y)
                # print('y.shape', y.shape)
                labels = self.knn.predict(y)
                # score = self.knn.predict_proba(y)
                # print('score',score)
                # print('label',label)
                for label in labels:
                    result += [[{
                        "score": 1,
                        "label": label
                    }]]
            except Exception as e:
                print('ERROR EVAL', e)
                traceback.print_exc()
                # result += [[{
                #     "score": 0,
                #     "label": -1
                # }]]
        return result
    

class ReportKNNHFDataset:
    def __init__(self, bs=128, fname=f'./_results/result_val.txt', mode='a', label='label'):
        self.label = label
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        self.fname = fname
        self.mode = mode
        self.bs = bs
    def evaluate(self, nets, net, netname, score, dataset, train_dataset):
        
        dot, gname = draw_stitchNet(nets, net, name=netname)
        
        task_evaluator = evaluator("image-classification")
        pipe = NetEvalPipeline(train_dataset, net, label=self.label)
        result = task_evaluator.compute(pipe, 
                       data=dataset,
                       metric="accuracy", input_column='pixel_values', label_column=self.label)
        accuracy = result['accuracy']

        # accuracy = accuracy_score_net_plants(net, dataset, train_dataset, bs=self.bs)
        
        print('accuracy', accuracy)
        macs, params = get_macs_params(net[0])
        network = net.get_id()
        self.f.write(f'{score},{accuracy},{macs},{params},{gname},"{network}"\n')
        self.f.flush()
        
    def close(self):
        self.f.close()
    
    def __enter__(self):
        self.f = open(self.fname, self.mode)
        return self
  
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        self.close()
        return True
        