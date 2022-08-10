from stitchnet.stitchonnx.utils import load_onnx_model, PROVIDERS, load_cats_and_dogs_dset, load_dl, convert_imagenet_to_cat_dog_label, change_input_dim
import onnxruntime as ort
from tqdm import tqdm
import torch
import numpy as np

def get_accuracy_outputs(onnx_model, dataset, bs=64):
    change_input_dim(onnx_model)
    ort_sess1 = ort.InferenceSession(onnx_model.SerializeToString(), providers=PROVIDERS)
    count = 0
    ys = []
    for x,t in tqdm(torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)):
        inputs = {}
        inputs[onnx_model.graph.input[0].name] = x.numpy()
        outputs = ort_sess1.run(None, inputs)
        y = outputs[0]
        ys.append(y)
        y = np.argmax(y, 1)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy, np.vstack(ys)

def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def get_ensemble_accuracy(onnx_model_paths, dataset, batch_size=64, geometric=True):
    individual_accs = []
    avg_ys = []
    for path in onnx_model_paths:
        onnx_model = load_onnx_model(path)
        acc,ys = get_accuracy_outputs(onnx_model, dataset, batch_size)
        individual_accs.append(acc)
        avg_ys.append(torch.nn.functional.softmax(torch.from_numpy(ys)))

    if geometric:
        avg_y = gmean(torch.stack(avg_ys), 0)
    else:
        avg_y = torch.mean(torch.stack(avg_ys), 0)
    
    ts = []
    for _,t in tqdm(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)):
        ts += list(t.numpy())
    ts = np.array(ts)
    ys = avg_y.argmax(1)
    ys = ys.numpy()
    ys = convert_imagenet_to_cat_dog_label(ys)
    accuracy = 1.*np.sum(ys == ts)/len(ts)
    return accuracy, individual_accs