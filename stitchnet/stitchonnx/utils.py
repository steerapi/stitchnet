from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
from skl2onnx.proto.onnx_helper_modified import (
    make_node, make_tensor_value_info, make_graph,
    make_model, ValueInfoProto
)
from onnx import helper
import onnxoptimizer
import onnx, onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import numpy as np
import traceback
import torch
import time
import copy
from functools import reduce
import operator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    torch.ones(1).cuda()
except:
    device = "cpu"

# PROVIDERS = ['CPUExecutionProvider']
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# PROVIDERS = ['CUDAExecutionProvider']

# def _create_onnx_model(model):
#     graph = make_graph(model.graph.node, model.graph.name, model.graph.input,
#                        model.graph.output, model.graph.initializer)
#     onnx_model = make_model(graph)
#     onnx_model.ir_version = model.ir_version
#     onnx_model.producer_name = model.producer_name
#     onnx_model.producer_version = model.producer_version
#     onnx_model.domain = model.domain
#     onnx_model.model_version = model.model_version
#     onnx_model.doc_string = model.doc_string
#     if len(model.metadata_props) > 0:
#         values = {p.key: p.value for p in model.metadata_props}
#         onnx.helper.set_model_props(onnx_model, values)

#     if len(onnx_model.graph.input) != len(model.graph.input):
#         raise RuntimeError("Input mismatch {} != {}".format(
#             len(onnx_model.input), len(model.input)))

#     # fix opset import
#     del onnx_model.opset_import[:]
#     for oimp in model.opset_import:
#         op_set = onnx_model.opset_import.add()
#         op_set.domain = oimp.domain
#         op_set.version = oimp.version
#     return onnx_model

def select_model_inputs_outputs2(model, outputs=None, inputs=None, inputs_types=None, inputs_shapes=None, name=None):
    """
    Takes a model and changes its outputs.

    :param model: *ONNX* model
    :param inputs: new inputs
    :param outputs: new outputs
    :return: modified model

    The function removes unneeded files.
    """
    # if inputs is not None:
    #     raise NotImplementedError("Parameter inputs cannot be empty.")
    if outputs is None:
        raise RuntimeError("Parameter outputs cannot be None.")
    if not isinstance(outputs, list):
        outputs = [outputs]
    if inputs is not None and not isinstance(inputs, list):
        inputs = [inputs]
    if inputs_types is None:
        inputs_types = []
    elif not isinstance(inputs_types, list):
        inputs_types = [inputs_types]
    if inputs_shapes is None:
        inputs_shapes = []
    elif not isinstance(inputs_shapes, list):
        inputs_shapes = [inputs_shapes]
        
    mark_var = {}
    for out in enumerate_model_node_outputs(model):
        mark_var[out] = 0
    for inp in model.graph.input:
        mark_var[inp.name] = 0
    for out in outputs:
        if out not in mark_var:
            raise ValueError("Output '{}' not found in model.".format(out))
        mark_var[out] = 1
    
    mark_var_inp = {}
    if inputs is not None:
        for inp in inputs:
            # if inp not in mark_var_inp:
            #     raise ValueError("Input '{}' not found in model.".format(inp))
            mark_var_inp[inp] = 1
        
    nodes = model.graph.node[::-1]
    mark_op = {}
    for node in nodes:
        mark_op[node.name] = 0

    # We mark all the nodes we need to keep.
    nb = 1
    while nb > 0:
        nb = 0
        for node in nodes:
            # decide whether to include this node
            if mark_op[node.name] == 1:
                continue
            mod = False
            # if output of this node is in the graph, include the node and check its input as well
            for out in node.output:
                if mark_var[out] == 1:
                    mark_op[node.name] = 1
                    mod = True
                    break
            if not mod:
                continue

            nb += 1
            for inp in node.input:
                # stop
                if mark_var_inp.get(inp, 0) == 1:
                    mark_op[node.name] = 1
                    continue
                if mark_var.get(inp, 0) == 1:
                    continue
                mark_var[inp] = 1
                nb += 1
        
    # All nodes verifies mark_op[node.name] == 1
    keep_nodes = [node for node in nodes if mark_op[node.name] == 1]
    keep_nodes = keep_nodes[::-1]

    var_out = []
    for out in outputs:
        value_info = ValueInfoProto()
        value_info.name = out
        var_out.append(value_info)
            
    if inputs is None:
        var_int = model.graph.input
    else:
        var_int = []
        for i,inp in enumerate(inputs):
            ttype = inputs_types[i]
            tshape = inputs_shapes[i]
            var_int.append(helper.make_tensor_value_info(inp, 
                                  ttype, tshape))
    if name is None:
        name = model.graph.name
    graph = make_graph(keep_nodes, name, var_int,
                       var_out, model.graph.initializer)
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    # if len(onnx_model.graph.input) != len(model.graph.input):
    #     raise RuntimeError("Input mismatch {} != {}".format(
    #         len(onnx_model.input), len(model.input)))

    # fix opset import
    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
        
    # passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    passes = ["eliminate_unused_initializer"]
    # print('inputname', [n.name for n in onnx_model.graph.input], 'inputs', inputs)
    # list_ops(onnx_model)
    # onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    # save_onnx_model(onnx_model, 'tmp.onnx')
    # onnx_model = load_onnx_model('tmp.onnx')
    
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)

    return optimized_model


def list_ops(model):
    for n in model.graph.node:
        print(n.op_type, n.input, n.output, n.name)

def get_numpy_matrix(model_onnx, name):
    [tensor] = [t for t in model_onnx.graph.initializer if t.name == name]
    w = numpy_helper.to_array(tensor)
    return w

def to_tensor_proto(array, name=None):
    tensor_proto = numpy_helper.from_array(array, name=name)
    return tensor_proto

def split_model_at(model_onnx, output, x):
    model_onnx1 = select_model_inputs_outputs2(model_onnx, output)
    
    # providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_sess1 = ort.InferenceSession(model_onnx1.SerializeToString(), providers=PROVIDERS)
    inputs = {}
    inputs[model_onnx1.graph.input[0].name] = x
    outputs1 = ort_sess1.run(None, inputs)
    
    model_onnx2 = select_model_inputs_outputs2(model_onnx, [o.name for o in model_onnx.graph.output], output, 1, outputs1[0].shape)

    return model_onnx1, model_onnx2, outputs1

def remove_dups(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def find_conv_inputs(model_onnx):
    inputs = []
    for n in model_onnx.graph.node:        
        if n.op_type == 'Conv':
            groups = [helper.get_attribute_value(att) for att in n.attribute if att.name == 'group' and helper.get_attribute_value(att) > 1]
            ngroups = len(groups)
            if ngroups == 0:
                inputs.append(n.input[0])
        elif n.op_type == 'Gemm':
            inputs.append(n.input[0])
    return inputs
    # return remove_dups(inputs)

def create_onnx_model(model, nodes=None, name=None, inputs=None, outputs=None, initializer=None):
    if nodes is None:
        nodes = model.graph.node
    if name is None:
        name = model.graph.name
    if inputs is None:
        inputs = model.graph.input
    if outputs is None:
        outputs = model.graph.output
    if initializer is None:
        initializer = model.graph.initializer
    
    graph = make_graph(nodes, name, inputs,
                       outputs, initializer)
    
    onnx_model = make_model(graph)
    onnx_model.ir_version = model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    # fix opset import
    del onnx_model.opset_import[:]
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version
        
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    # optimized_model = onnx_model

    return optimized_model

def replace_w(model_onnx, name, w):
    initializer = [] 
    for init in model_onnx.graph.initializer:
        if init.name == name:
            initializer.append(to_tensor_proto(w, name=name))
        else:
            initializer.append(init)        
    return create_onnx_model(model_onnx, initializer=initializer)

def replace_ws(model_onnx, names, ws):
    initializer = [] 
    for init in model_onnx.graph.initializer:
        if init.name in names:
            i = names.index(init.name)
            w = ws[i]
            name = names[i]
            initializer.append(to_tensor_proto(w, name=name))
        else:
            initializer.append(init)        
    return create_onnx_model(model_onnx, initializer=initializer)

def get_fragments(model, x, n=3):
    '''chop up onnx models into fragments at conv and linear(gemm) layers'''
    inputs = find_conv_inputs(model)
    inputs = inputs[1::n]
    fragments = []
    # print('[POTENTIAL SPLIT LOCATIONS]:', inputs)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    x = [x]
    for i,inp in enumerate(inputs):
        # print(i, inp)
        try:
            newf1, newmodel, newx = split_model_at(model, inp, x[0])
            # print('len(get_input_nodes(newf1))', len(get_input_nodes(newmodel)), [n.name for n in get_input_nodes(newmodel)])
            inpnodes = get_input_nodes(newmodel)
            if len(newf1.graph.node)==0:
                continue
            # print('inputnodes', i, inp, len(inpnodes), [n.op_type for n in inpnodes], [n.input for n in inpnodes])
            if len(inpnodes) > 1 and not reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]):
                continue
            f1 = newf1
            model = newmodel
            x = newx
            change_input_dim(f1)
            # print(input_name, f1.graph.input[0].name)
            if f1.graph.input[0].name == input_name:
                change_name(f1, "start")
            fragments.append(f1)
        except Exception as e:
            # skip multiple input dependencies
            # print('[WARNING]:', i, inp, e)
            # traceback.print_exc()
            pass
    # print('len(get_input_nodes(model))', len(get_input_nodes(model)), [n.name for n in get_input_nodes(model)])
    inpnodes = get_input_nodes(model)
    # print('inputnodes', inp, len(inpnodes), [n.op_type for n in inpnodes], [n.input for n in inpnodes])
    if len(inpnodes) > 1 and not reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]):
        return fragments
    change_input_dim(model)
    if model.graph.output[0].name == output_name:
        change_name(model, "end")
    fragments.append(model)
    return fragments

def execute_fragments(fragments, x):
    '''get outputs of all fragments'''
    outputs = [None]*len(fragments)
    
    for i,f in enumerate(fragments):
        change_input_dim(f)
        # print('execute_fragments', 'change input', f.graph.input)
        ort_sess = ort.InferenceSession(f.SerializeToString(), providers=PROVIDERS)
        inputs = {}
        # print("exec", "input", f.graph.input[0].name, f.graph.input[0].type)
        # print(x.shape)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        inputs[f.graph.input[0].name] = x
        o = ort_sess.run(None, inputs)
        x = o[0]
        outputs[i] = x
    return outputs

from collections import defaultdict
import hashlib
# print(hashlib.md5("whatever your string is".encode('utf-8')).hexdigest())
def hash_model(f):
    change_input_dim(f)
    hashf = hashlib.md5(f.SerializeToString()).hexdigest()
    return hashf

from .viz import draw_net

class Net:
    def __init__(self, fragments, nId=None):
        self.id = nId
        self.fragments = fragments
        # for f in fragments:
        #     hashf = hash_model(f)
        #     print(hashf)
        fragmentCs = []
        for fId,f in enumerate(fragments):
            change_input_dim(f)
            fragmentC = Fragment(f, self, fId)
            fragmentCs.append(fragmentC)
        self.fragmentCs = fragmentCs
        self.results = defaultdict(dict)
        self.knn = None
        self.p = None
    
    def fit(self, train_dataset, label_column="label", batch_size=32):
        label = label_column
        net = self
        
        fragmentC = net[0]
        change_input_dim(fragmentC.fragment)
        self.ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
        self.input_name = fragmentC.fragment.graph.input[0].name
        
        nnX = []
        nnY = []
        ptdset_train = TensorDataset(train_dataset['pixel_values'], 
                             train_dataset[label])
        dl = load_dl(ptdset_train, batch_size=batch_size, shuffle=False, num_workers=0)
        for dataItem in tqdm(dl):
            try:
                X = dataItem[0].squeeze(1).numpy()
                t = dataItem[1].unsqueeze(1).numpy()
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
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(nnX, nnY)
        
        self.knn = knn
        self.task = "image-classification"
    
    def __call__(self, pixel_values, **kwargs):
        return self.predict(pixel_values)
    def predict(self, pixel_values):
        result = []
        for pixel_value in tqdm(load_dl(pixel_values, batch_size=32, shuffle=False, num_workers=0)):
        # for pixel_value in tqdm(pixel_values):
            try:
                X = pixel_value.squeeze(1).numpy()
                inputs = {}
                inputs[self.input_name] = X
                outputs = self.ort_sess1.run(None, inputs)
                y = outputs[0]
                y = np.array(y)
                # print('y.shape', y.shape)
                labels = self.knn.predict(y)
                scores = self.knn.predict_proba(y)
                # print('score',score)
                # print('label',label)
                for score,label in zip(scores,labels):
                    result += [{
                        "score": list(score),
                        "label": label
                    }]
            except Exception as e:
                print('ERROR EVAL', e)
                traceback.print_exc()
                result += [{
                    "score": 0,
                    "label": -1
                }]
        return result
    
    def evaluate_dataset(self, dataset_val, label_column='labels'):
        result = self(dataset_val['pixel_values'])
        total = len(dataset_val)
        count = 0
        for r,t in zip(result,dataset_val['labels']):
            if r['label']==t:
                count+=1
        return {'accuracy': 1.*count/total}  
        
    def predict_files(self, filenames):
        if not isinstance(filenames, list):
            filenames = [filenames]            
        if self.knn is None:
            raise Exception("classifier is not trained, please call fit before")
        if self.p is None:
            self.p = AutoProcessor.from_pretrained('microsoft/resnet-50')
        result = []
        try:
            from PIL import Image
            data = []
            for filename in filenames:
                im = Image.open(filename)
                data += self.p(im)['pixel_values']
            X = np.stack(data)
            inputs = {}
            inputs[self.input_name] = X
            outputs = self.ort_sess1.run(None, inputs)
            y = outputs[0]
            y = np.array(y)
            # print('y.shape', y.shape)
            labels = self.knn.predict(y)
            scores = self.knn.predict_proba(y)
            for score,label in zip(scores,labels):
                result += [{
                    "score": list(score),
                    "label": label
                }]
        except Exception as e:
            print('ERROR EVAL', e)
            traceback.print_exc()
            result += [{
                "score": 0,
                "label": -1
            }]
        return result
        
    def draw_svg(self, path):
        draw_net(self, path)
        
    def save_onnx(self, path):
        self.save(path)
        
    def save(self, path):
        # print('len(self.fragments)', len(self.fragments))
        if len(self.fragments) == 1:
            print('saving to', path)
            save_onnx_model(self.fragments[0], f'{path}.onnx')
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            for i,fragment in enumerate(self.fragments):
                save_onnx_model(fragment, f'{path}_{i:03}.onnx')
        
    def get_id(self):
        return self.id
    def get_scores(self, x1, data, scoring_method='CKA'):
        tX = torch.from_numpy(x1)
        score_fragments = []
        for i,f in enumerate(self.fragments[:-1]):
            x2 = self.get_output(f,data)
            tY = torch.from_numpy(x2)
            score = get_score(tX, tY, min(tX.shape[1],tY.shape[1])*10, scoring_method)
            score_fragments.append((score, self.fragmentCs[i+1]))
        return score_fragments
    
    def evaluate(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        hashx = hashlib.md5(x.tobytes()).hexdigest()
        os = execute_fragments(self.fragments, x)
        for f,o in zip(self.fragments,os):
            hashf = hash_model(f)
            # print(hashf, hashx)
            self.results[hashf][hashx] = o
    
    def get_outputs(self, x):
        return [self.get_output(f,x) for f in self.fragments]
    
    def get_output(self, f, x):
        hashf = hash_model(f)
        # print('hashf', hashf)
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        hashx = hashlib.md5(x.tobytes()).hexdigest()
        if hashx in self.results[hashf]:
            return self.results[hashf][hashx]
        else:
            self.evaluate(x)
            return self.results[hashf][hashx]
        
    def get_input(self, f, x):
        curr = None
        for nf in self.fragments:
            prev = curr
            curr = nf
            if hash_model(nf) == hash_model(f):
                if prev is None:
                    return x
                return self.get_output(prev, x)
            
    def __iter__(self):
        self.index = 0
        return iter(self.fragmentCs)
    # def __next__(self):
    #     if self.index < len(self):
    #         item = self.fragmentCs[self.index]
    #         self.index+=1
    #         return item
    #     else:
    #         raise StopIteration
    def __getitem__(self,i):
        return self.fragmentCs[i]
    def __len__(self):
        return len(self.fragmentCs)
        
def get_input_nodes(f):
    # print('f.graph.input', [n.name for n in f.graph.input])
    nodes = []
    for n in f.graph.node:
        for inp in f.graph.input:
            if inp.name in n.input:
                nodes.append(n)
    return nodes
    
def get_output_nodes(f):
    # print('f.graph.output', [n.name for n in f.graph.output])
    nodes = []
    for n in f.graph.node:
        for out in f.graph.output:
            if out.name in n.output:
                nodes.append(n)
    return nodes


class Fragment: 
    def __init__(self, fragment, net: Net, fId=None):
        self.id = fId
        self.fragment = fragment
        self.net = net
    def get_last_id(self):
        if type(self.net.id) is tuple:
            return self.net.id[-1]
        else:
            return self.get_id()
    def get_id(self):
        return (self.net.id,self.id)
    def get_output(self, x):
        return self.net.get_output(self.fragment, x)
    def get_input(self, x):
        return self.net.get_input(self.fragment, x)
    def get_w(self):
        f = self.fragment
        
        nodes = get_input_nodes(f)
        
        node = nodes[0]
        name = node.input[1]
        w = get_numpy_matrix(f, name)
        return np.copy(w)
    
    def get_ws(self):
        f = self.fragment
        
        nodes = get_input_nodes(f)
        
        ws = []
        for node in nodes:
            name = node.input[1]
            w = get_numpy_matrix(f, name)
            ws.append(np.copy(w))
        return ws
    
    def replace_w(self, w, i=0):
        f = self.fragment
        nodes = get_input_nodes(f)
        node = nodes[i]
        name = node.input[1]
        return replace_w(f, name, w)
    
    def replace_ws(self, ws):
        f = self.fragment
        nodes = get_input_nodes(f)
        names = [node.input[1] for node in nodes]
        return replace_ws(f, names, ws)
    
    
def sample_index(X,Y, nsample=1000):
    num_samples = min(min(X.shape[0], Y.shape[0]), nsample)
    # print(X.shape, Y.shape)
    p = torch.ones(X.shape[0])
    indexX = p.multinomial(num_samples=num_samples)
    return indexX

# def change_node_name(node, name):
#     node.name = name
#     return node

# inplace op
def change_name(model, name):
    model.graph.name = name
    
# inplace op
def change_input_dim(model, dim="N"):
    sym_batch_dim = dim
    inputs = model.graph.input
    for input in inputs:
        dim1 = input.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim
    outputs = model.graph.output
    for output in outputs:
        if len(output.type.tensor_type.shape.dim)>0:
            dim1 = output.type.tensor_type.shape.dim[0]
            dim1.dim_param = sym_batch_dim

def adjust_w_linear(tX, tY, w):
    if tX.ndim == 2:
        acts1 = tX
        acts2 = tY
    else:
        pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flat = torch.nn.Flatten()
        tX = flat(pool(tX))

        acts1 = tX.reshape(tX.shape[0], -1)
        acts2 = tY
    
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    Ainit = acts2.T @ acts1.T.pinverse()
    # print(acts1.shape)
    # print(acts2.shape)
    # print('linear')
    A = train_w(acts1, acts2, Ainit)
    A = A.to(device)
    
    tw = torch.from_numpy(w).to(device)
    nw = torch.einsum('ij, jk -> ik', tw, A)
    nw = nw.cpu().numpy()
    return nw

from torch.autograd import Variable
def train_w(acts1, acts2, Winit, nepoch=1, batch_size=1024, learning_rate=1e-6, momentum=0.9):
    dtype = acts1.dtype
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    Winit = Winit.to(device)
    W = Variable(Winit, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=learning_rate, momentum=0.9)
    dset = torch.utils.data.TensorDataset(acts1,acts2)
    prev_loss = None
    for epoch in range(nepoch):
        running_loss = 0
        for x,y in torch.utils.data.DataLoader(dset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0):
            optimizer.zero_grad()
            y_pred = x.matmul(W.T)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        epoch_loss = running_loss/len(dset)
        # early stopping
        # print(f'epoch {epoch} loss', epoch_loss, acts1.shape, acts2.shape)
        if prev_loss is not None and prev_loss <= epoch_loss:
            break
        prev_loss = epoch_loss        
    return W.cpu().detach()

def adjust_w_conv(tX, tY, w):
    if tY.shape[-2]!=tX.shape[-2] and tY.shape[-1]!=tX.shape[-1]:
        up = torch.nn.UpsamplingBilinear2d((tY.shape[-2],tY.shape[-1]))
        tX = up(tX)
        # addLayers += [up]

    acts1 = tX.permute((0,2,3,1)).reshape(-1,tX.shape[1])
    acts2 = tY.permute((0,2,3,1)).reshape(-1,tY.shape[1])

    indexX = sample_index(acts1, acts2, nsample=min(acts1.shape[0], acts2.shape[0])*10)
    acts1sampled = acts1[indexX,:]
    acts2sampled = acts2[indexX,:]

    # print('diff sampled', (acts1sampled - acts2sampled).pow(2).sum())
    
    acts1 = acts1.to(device)
    acts2 = acts2.to(device)
    
    Ainit = acts2sampled.T @ acts1sampled.T.pinverse()
    # print(Ainit)
    # print(acts2.shape)
    A = train_w(acts1, acts2, Ainit)
    A = A.to(device)
    
    tw = torch.from_numpy(w).to(device)
    nw = torch.einsum('ijkl, jn -> inkl', tw, A)
    nw = nw.cpu().numpy()
    return nw

def adjust_w(tX, tY, w):
    if w.ndim == 2:
        return adjust_w_linear(tX, tY, w)
    else:
        # linear to conv, unsupport
        if tX.ndim == 2 and tY.ndim == 4:
            raise Exception("unsupport linear to conv stitching")
        # print('tX.shape', tX.shape, 'tY.shape', tY.shape)
        return adjust_w_conv(tX, tY, w)

# def clone_node(node):
#     newnode = helper.make_node(
#         node.op_type,                  # name
#         node.input, # inputs
#         node.output,                  # outputs
#         mode='constant',        # attributes
#     )
#     return newnode

def get_score_net_beans(net, data_score, num_samples=1000):
    totalscore = 1
    for i,fragment1 in tqdm(enumerate(net[:-1])):
        fragment2 = net[i+1]
        score = get_score_fragments(fragment1, fragment2, data_score, num_samples=num_samples)
        # print(fragment1.fragment.graph.output[0].name, fragment2.fragment.graph.input[0].name, score)
        totalscore *= score
    return totalscore

def get_score_net(net, data_score, num_samples=1000):
    totalscore = 1
    for i,fragment1 in tqdm(enumerate(net[:-1])):
        fragment2 = net[i+1]
        score = get_score_fragments(fragment1, fragment2, data_score, num_samples=num_samples)
        # print(fragment1.fragment.graph.output[0].name, fragment2.fragment.graph.input[0].name, score)
        totalscore *= score
    return totalscore

def get_data_score(batch_size=32, includeTarget=False):
    from stitchnet.stitchonnx.utils import load_cats_and_dogs_dset,convert_imagenet_to_cat_dog_label
    from stitchnet.stitchonnx.utils import accuracy_score_model,accuracy_score_net,load_dl
    from stitchnet.stitchonnx.utils import generate_networks, ScoreMapper
    from stitchnet.stitchonnx.report import Report
    from stitchnet.stitchonnx.utils import evalulate_stitchnet

    from tqdm import tqdm
    import torch
    import numpy as np
    import os
    from collections import defaultdict
    import hashlib
    import random
    import time

    random.seed(51)
    np.random.seed(24)
    torch.manual_seed(77)

    dataset_train = load_cats_and_dogs_dset("train")

    dl_score = load_dl(dataset_train, batch_size=batch_size)
    data_score,t = next(iter(dl_score))
    if includeTarget:
        return data_score,t
    data_score = data_score.numpy()
    return data_score

def get_score_fragments(fragment1: Fragment, fragment2: Fragment, data, num_samples=10000):
    x1 = fragment1.get_output(data)
    x2 = fragment2.get_input(data)
    tX = torch.from_numpy(x1)
    tY = torch.from_numpy(x2)
    score = get_score(tX, tY, num_samples=num_samples)
    return score

def stitch_all_fragments(net, data_score):
    curr = net[0]
    for i in range(1,len(net)):
        newcurr_fragment = stitch_fragments(curr, net[i], data_score)
        newcurr_net = Net([newcurr_fragment],i)
        curr = newcurr_net[0]
    return newcurr_net

def stitch_fragments(fragment1: Fragment, fragment2: Fragment, data):
    # list_ops(fragment1.fragment)
    # fragment1 = copy.deepcopy(fragment1)
    # fragment2 = copy.deepcopy(fragment2)
    
    try:
        x1 = fragment1.get_output(data)
        x2 = fragment2.get_input(data)
    except Exception as e:
        traceback.print_exc()
        # print('-------fragment1 ops-------')
        # list_ops(fragment1.fragment)
        # print('-------fragment2 ops-------')
        # list_ops(fragment2.fragment)
        raise e
    tX = torch.from_numpy(x1)
    tY = torch.from_numpy(x2)
    # score = get_score(tX, tY)
    # if score < 0.5:
    #     return None
    # print('score', score)
    
    ws = fragment2.get_ws()
    nws = []
    for i,w in enumerate(ws):
        nw = adjust_w(tX, tY, w)
        nws.append(nw)
    newFragment = fragment2.replace_ws(nws)
    # newFragment = fragment2.fragment
    
    oldinputname = newFragment.graph.input[0].name
    newname = oldinputname+"_timestamp_"+str((time.time()))
    
    # find and replace all the input name
    for n in newFragment.graph.node:
        for i,inp in enumerate(n.input):
            if inp == oldinputname:
                n.input[i] = newname
    newFragment.graph.input[0].name = newname
    
    # begin stitching
    exitNodes = get_output_nodes(fragment1.fragment)
    # print('exitNodes', [n.name for n in exitNodes])
    exitNode = exitNodes[0]

    nodes = []
    for node in fragment1.fragment.graph.node:
        if node.name == exitNode.name:
            # print('skipping...', node.name)
            continue
        node = copy.deepcopy(node)
        nodes.append(node)
    
    # newOutput = newname
    node = exitNode
    kwargs = {}
    for att in node.attribute:
        # print(att)
        kwargs[att.name] = helper.get_attribute_value(att)
        
        
    newNodeOutName = newname
    if tX.ndim == 4 and tY.ndim == 2:
        newNodeOutName += "_pool"
        poolNode = helper.make_node(
            'GlobalAveragePool',
            inputs=[f'{newNodeOutName}'],
            outputs=[f'{newNodeOutName}_poolflat'],
        )
        flatNode = helper.make_node(
            'Flatten',
            inputs=[f'{newNodeOutName}_poolflat'],
            outputs=[f'{newname}'],  # Default value for axis: axis=1
        )
        
    newNode = helper.make_node(
        node.op_type,                  # optype
        node.input, # inputs
        [newNodeOutName],               # outputs
        node.name+"_nodeexit_timestamp_"+str((time.time())), # name
        node.doc_string,
        node.domain,
        # mode='constant',        # attributes
        # name=newname+"_nodeexit_"+str((time.time()))
        # doc_string=exitNode.doc_string,
        # domain=exitNode.domain,
        # *node.attribute
        **kwargs
    )
    
    # newNode = helper.make_node(
    #     exitNode.op_type,                  # name
    #     exitNode.input, # inputs
    #     [newname],                  # outputs
    #     # mode='constant',        # attributes
    #     name=newname+"_nodeexit_"+str((time.time()))
    # )
    
    nodes.append(newNode)    
    if tX.ndim == 4 and tY.ndim == 2:
        nodes.append(poolNode)
        nodes.append(flatNode)
    
    inpnodes = get_input_nodes(newFragment)
    # print('[n.op_type for n in inputnodes]', [n.op_type for n in inpnodes])
    if len(inpnodes) > 1 and not reduce(operator.and_, ['Conv' == n.op_type for n in inpnodes]):
        raise Exception("There are more than one inputs to stitch and it is not all going into Conv.")

    if len(inpnodes) == 0:
        # print('len(inpnodes)', len(inpnodes))
        # print('newFragment', newFragment, fragment2.fragment)
        # list_ops('newFragment', newFragment)
        # list_ops('fragment2', fragment2.fragment)
        raise Exception("No inputs.")
        
    enterNode = inpnodes[0]
    
    node = enterNode
    kwargs = {}
    for att in node.attribute:
        # print(att)
        kwargs[att.name] = helper.get_attribute_value(att)
    
    newNode = helper.make_node(
        node.op_type,                  # optype
        [newname]+enterNode.input[1:], # inputs
        node.output,               # outputs
        node.name+"_nodeenter_timestamp_"+str((time.time())), # name
        node.doc_string,
        node.domain,
        # mode='constant',        # attributes
        # name=newname+"_nodeexit_"+str((time.time()))
        # doc_string=exitNode.doc_string,
        # domain=exitNode.domain,
        # *node.attribute
        **kwargs
    )
    
    # newNode = helper.make_node(
    #     enterNode.op_type,                  # name
    #     [newname]+enterNode.input[1:],    # inputs
    #     enterNode.output,              # outputs
    #     # mode='constant',        # attributes
    #     name=newname+"_nodeenter_"+str((time.time()))
    # )
    existingNames = {}
    for n in nodes:
        for item in n.input:
            existingNames[item] = True
        for item in n.output:
            existingNames[item] = True
    
    # change node names
    newnodes = []
    newnodes.append(newNode) 
    for node in newFragment.graph.node:
        if node.name == enterNode.name:
            # print('skipping...', node.name)
            continue
        node = copy.deepcopy(node)
        node.name += "_timestamp_"+str((time.time()))
        newnodes.append(node)
            
    initializer = []
    for init in fragment1.fragment.graph.initializer:
        initializer.append(init)
    
    newinitname = {}
    for init in newFragment.graph.initializer:
        init = copy.deepcopy(init)
        oldname = init.name
        init.name += "_timestamp_"+str((time.time()))
        newinitname[oldname] = init.name
        initializer.append(init)
    
    for node in newnodes:
        for i,inp in enumerate(node.input):
            if inp in existingNames:
                # print('existingNames', inp)
                if inp not in newinitname:
                    if 'timestamp' not in inp:
                        newinitname[inp] = inp+"_timestamp_"+str((time.time()))
                    # else:
                    #     newinitname[inp] = inp.split('_')[0]+"_timestamp_"+str((time.time()))
                    
                    # node.input[i] = newNames[inp]
                
        for i,out in enumerate(node.output):
            if out in existingNames:
                # print('existingNames', out)
                if out not in newinitname:
                    if 'timestamp' not in out:
                        newinitname[out] = out+"_timestamp_"+str((time.time()))
                    # else:
                    #     newinitname[out] = out.split('_')[0]+"_timestamp_"+str((time.time()))
                    # node.output[i] = newNames[out]
                # if inp not in newouts:
                #     newouts[inp] = inp+"_"+str((time.time()))
                # node.output[i] = newouts[inp]
    
    # replace input output with new names
    for node in newnodes:
        for i,inp in enumerate(node.input):
            if inp in newinitname:
                node.input[i] = newinitname[inp]
        for i,out in enumerate(node.output):
            if out in newinitname:
                node.output[i] = newinitname[out]
        if 'timestamp' not in node.name:
            node.name += "_timestamp_"+str((time.time()))
        else:
            node.name = node.name.split('_')[0]+"_timestamp_"+str((time.time()))
    
    inputs = copy.deepcopy(fragment1.fragment.graph.input)
    for inp in inputs:
        if inp.name in newinitname:
            inp.name = newinitname[inp.name]
    
    outputs = newFragment.graph.output
    for out in outputs:
        if out.name in newinitname:
            out.name = newinitname[out.name]
    
    nodes += newnodes
    # print('end stitching')
    name = 'stitch'
    try:
        newnet = create_onnx_model(fragment1.fragment, nodes=nodes, name=name, inputs=inputs, outputs=outputs, initializer=initializer)
        newnet = onnxoptimizer.optimize(newnet, passes = [
                                              'eliminate_nop_transpose',
                                              'eliminate_nop_pad',
                                              'fuse_consecutive_transposes',
                                              'fuse_transpose_into_gemm'
                                            ])
        change_input_dim(newnet)
        return newnet
    except Exception as e:
        print(e)
        traceback.print_exc()
        for node in nodes:
            print(node.name, node.input, node.output)
        
def list_net_ops(net):
    for f in net:
        list_ops(f.fragment)
        # print()

def get_all_fragments(nets):
    fragments = [f for net in nets for f in net]
    return fragments

# import multiprocessing as mp
# if __name__ == '__main__':
#     mp.set_start_method('spawn')
# from multiprocessing import Pool

# def net_scores(net2, x1, data):
#     return net2.get_scores(x1,data)
import threading

class ScoreMapper:
    def __init__(self, nets, data, scoring_method='CKA'):
        self.nets = nets
        self.scoreMap = {}
        self.data = data
        self.scoring_method = scoring_method
    def score(self, x1, curDepth, maxDepth=10, hashId=None):
        if isinstance(x1, torch.Tensor):
            x1 = x1.numpy()
        hashx = hashlib.md5(x1.tobytes()).hexdigest()
        # if hashId is None:
        # else:
        #     hashx = hashId
        # # print('hashx in self.scoreMap', hashx in self.scoreMap)
        # if hashx in self.scoreMap:
        #     return self.scoreMap[hashx]
        # with Pool(len(self.nets)) as p:
        #     nextscoresAll = p.map(net_scores, zip(self.nets, [x1]*5, [self.data]*5))
        # for nextscores in nextscoresAll:
        #     # only take ending segments
        #     if curDepth >= maxDepth-1:
        #         scores += nextscores[-1:]
        #     else:
        #         scores += nextscores
        
        scores = []
        def net_scores(net2, x1, data, scores):
            nextscores = net2.get_scores(x1,data,self.scoring_method)
            # only take ending segments
            if curDepth >= maxDepth-1:
                scores += nextscores[-1:]
            else:
                scores += nextscores        

        threads = [threading.Thread(
                    target=net_scores, 
                    args=(net,x1,self.data,scores)
              ) for net in self.nets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # scores = []
        # for net2 in tqdm(self.nets):
        #     nextscores = net2.get_scores(x1,self.data)
        #     # only take ending segments
        #     if curDepth >= maxDepth-1:
        #         scores += nextscores[-1:]
        #     else:
        #         scores += nextscores
        scores = sorted(scores, key=lambda x:x[0], reverse=True)
        # disabled caching for now since it is taking up too much memory
        # self.scoreMap[hashx] = scores
        # return self.scoreMap[hashx]
        return scores

def find_next_fragment(curr, scoreMapper, data, threshold=0.5, maxDepth=10, sample=False, K=None):
    x1 = curr.get_output(data)
    fId = curr.get_id()
    if type(fId[0]) is tuple:
        curDepth = len(fId[0])
    else:
        curDepth = 1
    # print('current depth:', curDepth) 
    scores = scoreMapper.score(x1, curDepth, maxDepth=maxDepth)
    if K is not None:
        scores = scores[:K]
    # print('potential next fragments:', len(scores))
    # print(f'potential next fragments before thresholding of {threshold}:', len(scores), [f'{s[0]:.2}' for s in scores])
    # filter out the previous in curr
    
    scores = [(s,nextf) for s,nextf in scores if not check_already_has(curr,nextf)]
    
    # print(f'potential next fragments after filter duplicated fragments:', len(scores), [f'{s[0]:.2}' for s in scores])
    print(f'potential next fragments before thresholding of {threshold}:', len(scores), [f'{s[0]:.2}' for s in scores])

    if threshold is not None:
        scores = [score for score in scores if score[0]>threshold]
    # print('scores', [f'{s[0]:.2}' for s in scores])
    if sample and len(scores)>0:
        i = np.random.choice(range(len(scores)))
        scores = [scores[i]]
    print(f'potential next fragments after thresholding of {threshold}:', len(scores), [f'{s[0]:.2}' for s in scores])
    for s,nextf in scores:
        # score = get_score_fragments(curr, nextf, data)
        # if score < threshold:
        #     continue
        # yield score,nextf
        # if s > threshold:
        #     yield s,nextf
        yield s,nextf

# check if nextf already in curr
def check_already_has(curr, nextf):
    fId = curr.get_id()
    nextFId = nextf.get_id()
    if type(fId[0]) is tuple:
        return nextFId in fId[0]
    else:
        return nextFId in (fId,)
    
def get_net_id(curr, nextf):
    fId = curr.get_id()
    if type(fId[0]) is tuple:
        nId = fId[0] + (nextf.get_id(),)
    else:
        nId = (fId,) + (nextf.get_id(),)
    return nId

import onnx_tool
from onnx_tool import create_ndarray_f32
from onnx_tool.node_profilers import graph_profile

def get_macs_params(fragment: Fragment, inputName=None, inputSize=(1, 3, 224, 224)):
    inputs= {}
    if inputName is None:
        inputName = fragment.fragment.graph.input[0].name
    inputs[inputName] = create_ndarray_f32(inputSize)
    # change_input_dim(fragment.fragment,1)
    macs,params,nodemap=graph_profile(fragment.fragment.graph, inputs, False)
    return macs, params

def get_macs_params_onnx(onnx_model, inputName=None, inputSize=(1, 3, 224, 224)):
    inputs= {}
    if inputName is None:
        inputName = onnx_model.graph.input[0].name
    inputs[inputName] = create_ndarray_f32(inputSize)
    # change_input_dim(fragment.fragment,1)
    macs,params,nodemap=graph_profile(onnx_model.graph, inputs, False)
    return macs, params

def recursive_stitching(curr, scoreMapper, data, threshold=0.9, totalThreshold=0.5, totalscore=1, maxDepth=10, sample=False, K=None):
    for score,nextf in find_next_fragment(curr, scoreMapper, data, threshold, maxDepth, sample, K):
        # skip fragment if it is the same as current fragment
        # print('curr', curr.fragment)
        # print('nextf', nextf.fragment)
        totalscore_nextf = totalscore*score
        print(f'totalscore before thresholding of {totalThreshold}: {totalscore_nextf}');
        if totalscore_nextf < totalThreshold:
            continue
        try:
            if nextf.fragment.graph.name=='end':
                newcurr_fragment = stitch_fragments(curr, nextf, data)
                newcurr_net = Net([newcurr_fragment], get_net_id(curr, nextf))
                newcurr = newcurr_net[0]
                yield totalscore_nextf, newcurr
            else:
                newcurr_fragment = stitch_fragments(curr, nextf, data)
                newcurr_net = Net([newcurr_fragment], get_net_id(curr, nextf))
                newcurr = newcurr_net[0]
                for _score, _curr  in recursive_stitching(newcurr, scoreMapper, data, threshold, totalThreshold, totalscore_nextf, maxDepth, sample, K):
                    yield _score, _curr
        except Exception as e:
            # catch death end path with errors
            print('[WARNING]', e)
            # traceback.print_exc()
            # raise e
            # pass
            
def generate_networks(nets, scoreMapper, data, threshold=0.9, totalThreshold=0.5, maxDepth=10, sample=False, K=None):
    fragments = [f for net in nets for f in net]
    starts = [f for f in fragments if f.fragment.graph.name == 'start']
    ends = [f for f in fragments if f.fragment.graph.name == 'end']
    middles = [f for f in fragments if f.fragment.graph.name not in ['start','end']]
    if sample:
        starts = [np.random.choice(starts)]
    for start in starts:
        for score, curr in recursive_stitching(start, scoreMapper, data, threshold, totalThreshold, 1.0, maxDepth=maxDepth, sample=sample, K=K):
            yield score, curr.net
        
# def sample_network(nets, data, maxround=10, threshold=0.5):
#     fragments = [f for net in nets for f in net]
#     starts = [f for f in fragments if f.fragment.graph.name == 'start']
#     ends = [f for f in fragments if f.fragment.graph.name == 'end']
#     middles = [f for f in fragments if f.fragment.graph.name not in ['start','end']]
    
#     print('len(starts)', len(starts))
#     print('len(middles)', len(middles))
#     print('len(ends)', len(ends))
    
#     start = np.random.choice(starts)
#     curr = start
#     totalscore = 1
#     while maxround >= 0:
#         nextf = np.random.choice(middles + ends)
#         score = get_score_fragments(curr, nextf, data)
#         if score < threshold:
#             continue
#         try:
#             curr_fragment = stitch_fragments(curr, nextf, data)
#             curr_net = Net([curr_fragment])
#             curr = curr_net[0]
#             totalscore *= score
#             if nextf.fragment.graph.name == 'end':
#                 break
#         except:
#             # traceback.print_exc()
#             pass
#         maxround-=1
        
#     if nextf.fragment.graph.name != 'end':
#         nextf = np.random.choice(ends)
#         # print('pickend')
#         # list_ops(nextf.fragment)
#         curr_fragment = stitch_fragments(curr, nextf, data)
#         curr_net = Net([curr_fragment])
#         score = get_score_fragments(curr, nextf, data)
#         totalscore *= score
#     return curr_net, totalscore
        
def evalulate_stitchnet(net, data):
    import onnxruntime as ort
    data = data if type(data).__module__ == np.__name__ else data.numpy()
    fragmentC = net[0]
    change_input_dim(fragmentC.fragment)
    ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
    inputs = {}
    inputs[fragmentC.fragment.graph.input[0].name] = data
    outputs = ort_sess1.run(None, inputs)
    data = outputs[0]
        # print(data.shape)
    return data
    
from .ptCKA import linear_CKA
    
def pt_R2(X, Y):
    # QR decomposition of Y
    Q, R = torch.linalg.qr(Y)

    # Compute the Frobenius norm of Q^T * X
    # print('Q', Q.shape)
    # print('X', X.shape)
    # print('Q^T * X', torch.matmul(Q.T, X).shape)
    numerator = torch.norm(torch.matmul(Q.T, X), p='fro') ** 2
    # Compute the Frobenius norm of X
    denominator = torch.norm(X, p='fro') ** 2
    # print('numerator', numerator.item(), 'denominator', denominator.item(), 'numerator / denominator', numerator.item() / denominator.item())

    # Compute and return R^2_LR
    r2 = (numerator / denominator)
    # if r2 >= 1:
    #     print('X',X)
    #     print('Y',Y)
    # print('r2', r2.item(), numerator.item(), denominator.item())
    return r2

@torch.no_grad()
def get_score(X, Y, num_samples=1000, scoring_method='CKA'):
    if scoring_method == 'R2':
        score_function = pt_R2
    if scoring_method == 'CKA':
        score_function = linear_CKA
    
    # X = torch.from_numpy(X)
    # Y = torch.from_numpy(Y)
    # print('X1', X.shape)
    # print('Y1', Y.shape)
    X = X.to(device)
    Y = Y.to(device)
    if X.ndim == 4 and Y.ndim == 4:
        if X.shape[2] > Y.shape[2]:
            up = torch.nn.UpsamplingBilinear2d((Y.shape[-2],Y.shape[-1]))
            X = up(X)
        else:
            up = torch.nn.UpsamplingBilinear2d((X.shape[-2],X.shape[-1]))
            Y = up(Y)
        # print('X2', X.shape)
        # print('Y2', Y.shape)
        X = X.permute((0,2,3,1)).reshape(-1,X.shape[1])
        Y = Y.permute((0,2,3,1)).reshape(-1,Y.shape[1])
        # only sampling 10 times the channels
        num_samples = min(num_samples, X.shape[0], Y.shape[0])
        # print(num_samples, X.shape, Y.shape)
        p = torch.ones(X.shape[0])
        indexX = p.multinomial(num_samples=num_samples)
        # p = torch.ones(Y.shape[0])
        # indexY = p.multinomial(num_samples=num_samples)
        # print(X[indexX,:].shape, Y[indexY,:].shape)
        s = score_function(X[indexX,:],Y[indexX,:]).cpu().item()
    elif X.ndim == 4 and Y.ndim == 2:
        pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flat = torch.nn.Flatten()
        X = flat(pool(X))
        b = X.shape[0]
        X = X.reshape(b,-1)
        Y = Y.reshape(b,-1)
        num_samples = min(num_samples,X.shape[1],Y.shape[1])
        p = torch.ones(X.shape[1])
        indexX = p.multinomial(num_samples=num_samples)
        p = torch.ones(Y.shape[1])
        indexY = p.multinomial(num_samples=num_samples)
        X = X[:,indexX]
        Y = Y[:,indexY]
        s = score_function(X,Y).cpu().item()
    else:
        # TODO: check b > 2*min(X.shape[1],Y.shape[1])
        b = X.shape[0]
        X = X.reshape(b,-1)
        Y = Y.reshape(b,-1)
        num_samples = min(num_samples,X.shape[1],Y.shape[1])
        p = torch.ones(X.shape[1])
        indexX = p.multinomial(num_samples=num_samples)
        p = torch.ones(Y.shape[1])
        indexY = p.multinomial(num_samples=num_samples)
        X = X[:,indexX]
        Y = Y[:,indexY]
        s = score_function(X,Y).cpu().item()
        
    # CKA = linear_CKA(X, Y).cpu().item()
    # CKA = linear_CKA(X.cuda(), Y.cuda()).cpu().item()
    return s

from tqdm import tqdm

catIds = [281,282,283,284,285]
dogIds = list(range(151,251))
def convert_imagenet_to_cat_dog_label(y):
    ys = []
    for py in y:
        if py in catIds:
            py = 0
        elif py in dogIds:
            py = 1
        else:
            py = -1
        ys.append(py) 
    return np.array(ys)

@torch.no_grad()
def accuracy_score_model(model, dataset, bs=64, num_workers=6):
    count = 0
    model.eval()
    for x,t in tqdm(load_dl(dataset, batch_size=bs, shuffle=False, num_workers=num_workers)):
        x = x.to(device)
        model.to(device)
        y = model(x)
        y = y.cpu()
        y = np.argmax(y.detach().numpy(), 1)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy

def accuracy_score_net_old(net, dataset, bs=64, num_workers=6):
    assert len(net)==1, 'stitchnet should have only one fragment'
    count = 0
    for x,t in tqdm(load_dl(dataset, batch_size=bs, shuffle=False, num_workers=num_workers)):
        y = evalulate_stitchnet(net, x)
        # y = ys[0]
        # print('y.shape', y.shape)
        y = np.argmax(y, 1)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy

import onnxruntime as ort

def accuracy_score_net(net, dataset, bs=64, num_workers=6):
    assert len(net)==1, 'stitchnet should have only one fragment'
    count = 0
    fragmentC = net[0]
    change_input_dim(fragmentC.fragment)
    ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
    
    for x,t in tqdm(load_dl(dataset, batch_size=bs, shuffle=False, num_workers=num_workers)):
        data = x.numpy()

        # y = evalulate_stitchnet(net, x)
        inputs = {}
        inputs[fragmentC.fragment.graph.input[0].name] = data
        outputs = ort_sess1.run(None, inputs)
        y = outputs[0]
        # y = ys[0]
        # print('y.shape', y.shape)
        y = np.argmax(y, 1)
        # print('y.shape', y.shape)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy

from sklearn.neighbors import KNeighborsClassifier
def accuracy_score_net_plants(net, dataset, train_dataset, bs=64, num_workers=6):
    assert len(net)==1, 'stitchnet should have only one fragment'
    count = 0
    fragmentC = net[0]
    change_input_dim(fragmentC.fragment)
    ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
    
    
    # train the classfier first
    nnX = []
    nnY = []
    ptdset_train = TensorDataset(train_dataset['pixel_values'], 
                             train_dataset['labels'])
    dl = load_dl(ptdset_train, batch_size=bs, shuffle=False, num_workers=0)
    for dataItem in tqdm(dl):

    # for x,t in tqdm(load_dl(train_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)):
        # data = x.numpy()
        data = dataItem[0].squeeze(1).numpy()
        t = dataItem[1].unsqueeze(1).numpy()

        # y = evalulate_stitchnet(net, x)
        inputs = {}
        inputs[fragmentC.fragment.graph.input[0].name] = data
        outputs = ort_sess1.run(None, inputs)
        y = outputs[0]
        nnX.append(y)
        nnY.append(t)
        
    nnX = np.vstack(nnX)
    nnY = np.vstack(nnY).squeeze()
    # print(nnX.shape, nnY.shape)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(nnX, nnY)
    
    # evaluate 
    # for x,t in tqdm(load_dl(dataset, batch_size=bs, shuffle=False, num_workers=num_workers)):
    #     data = x.numpy()
    ptdset_test = TensorDataset(dataset['pixel_values'], 
                             dataset['labels'])
    dl = load_dl(ptdset_test, batch_size=bs, shuffle=False, num_workers=0)
    for dataItem in tqdm(dl):

    # for x,t in tqdm(load_dl(train_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)):
        # data = x.numpy()
        data = dataItem[0].squeeze(1).numpy()
        t = dataItem[1].numpy()
        # print("t", t)

        # y = evalulate_stitchnet(net, x)
        inputs = {}
        inputs[fragmentC.fragment.graph.input[0].name] = data
        outputs = ort_sess1.run(None, inputs)
        y = outputs[0]
        # print("y",y)
        y = knn.predict(y)
        
        count += np.sum(y == t)
        # print("y, t", y, t, np.sum(y == t))
    # print(len(ptdset_test))
    accuracy = 1.*count/len(ptdset_test)
    return accuracy

# data cat and dog
# https://www.kaggle.com/competitions/dogs-vs-cats/rules

import torchvision
import torch
from torchvision.models import ResNet50_Weights
import os
from pathlib import Path


from torch.utils.data import TensorDataset
from transformers import AutoProcessor
from datasets import load_dataset
def load_hf_train_val_dset(name='sampath017/plants', test_fraction=0.2, seed=47):
    dataset = load_dataset(name)
    dset = dataset["train"].train_test_split(test_fraction, seed=seed)
    p = AutoProcessor.from_pretrained('microsoft/resnet-50')
    def processor_function(examples):
        return p(examples["image"])
    dset = dset.map(processor_function, batched=True)    
    small_train_dataset = dset['train']
    small_eval_dataset = dset['test']
    dataset_train = TensorDataset(torch.Tensor(small_train_dataset['pixel_values']), torch.tensor(small_train_dataset['label']))
    dataset_val = TensorDataset(torch.Tensor(small_eval_dataset['pixel_values']), torch.tensor(small_eval_dataset['label']))
    return dataset_train, dataset_val

def load_hf_train_val_dset_with_test_split(name='food101', train='train', val='validation',label="labels", num_train=None, num_val=None, streaming=False, seed=47):
    dataset = load_dataset(name, streaming=streaming)
    dset_train = dataset[train]
    p = AutoProcessor.from_pretrained('microsoft/resnet-50')
    def processor_function(examples):
        try:
            r = p(examples["image"])
            return r
        except:
            return {'pixel_values': None}
    
    if num_train is not None:
        if streaming:
            dset_train = dset_train.shuffle(seed=seed).take(num_train)
        else:
            dset_train = dset_train.shuffle(seed=seed).select(range(num_train))
    else:
        dset_train = dset_train.shuffle(seed=seed)
        
    small_train_dataset = dset_train.map(processor_function, batched=False).filter(lambda x:x['pixel_values'] is not None, batched=False)   
    small_train_dataset = small_train_dataset.with_format(type='torch')
        
    if val is not None:
        dset_test = dataset[val]
        if num_val is not None:
            if streaming:
                dset_test = dset_test.shuffle(seed=seed).take(num_val)
            else:
                dset_test = dset_test.shuffle(seed=seed).select(range(num_val))
        else:
            dset_test = dset_test.shuffle(seed=seed)

        small_test_dataset = dset_test.map(processor_function, batched=False).filter(lambda x:x['pixel_values'] is not None, batched=False)
        small_test_dataset = small_test_dataset.with_format(type='torch')

        return small_train_dataset, small_test_dataset
    else:
        return small_train_dataset

def load_cats_and_dogs_dset(folder="train"):
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    dataset = torchvision.datasets.ImageFolder(f'_data/dogs_cats/{folder}', transform=preprocess)
    return dataset

def load_dl(dset, collate_fn=None, batch_size=64, shuffle=True, num_workers=6):
    dl = torch.utils.data.DataLoader(dset, collate_fn=collate_fn, shuffle=shuffle, batch_size=batch_size)
    return dl

# def load_cats_and_dogs():
#     weights = ResNet50_Weights.IMAGENET1K_V1
#     preprocess = weights.transforms()
#     dataset = torchvision.datasets.ImageFolder(Path(os.path.dirname(__file__)+'/../_data/dogs_cats/train').resolve(), transform=preprocess)
#     dl = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64)
#     return dl


# def find_network(nets, data, maxround=10):
#     fragments = [f for net in nets for f in net]
#     starts = [f for f in fragments if f.fragment.graph.name == 'start']
#     ends = [f for f in fragments if f.fragment.graph.name == 'end']
#     middles = [f for f in fragments if f.fragment.graph.name not in ['start','end']]
    
#     start = np.random.choice(starts)
#     curr = start
#     totalscore = 1
#     while maxround >= 0:
#         nextf = np.random.choice(middles + ends)
#         score = get_score_fragments(curr, nextf, data)
#         if score < threshold:
#             continue

#         # print('pick')
#         # list_ops(nextf.fragment)
#         # if get_score_fragments(curr, nextf, data) < 0.1:
#         #     continue
#         try:
#             curr_fragment = stitch_fragments(curr, nextf, data)
#             curr_net = Net([curr_fragment])
#             curr = curr_net[0]
#             totalscore *= score
#             # print('c', curr)
#             if nextf.fragment.graph.name == 'end':
#                 break
#         except:
#             # traceback.print_exc()
#             pass
#         maxround-=1
        
#     if nextf.fragment.graph.name != 'end':
#         nextf = np.random.choice(ends)
#         # print('pickend')
#         # list_ops(nextf.fragment)
#         curr_fragment = stitch_fragments(curr, nextf, data)
#         curr_net = Net([curr_fragment])
#         score = get_score_fragments(curr, nextf, data)
#         totalscore *= score
#     return curr_net, totalscore
        
    
from collections import defaultdict
from functools import reduce

@torch.no_grad()
def eval_original_model(model, dataloaders):
    model.to(device)
    model.eval()
    
    epoch_acc = defaultdict(int)
    for phase in ['train', 'val']:
        running_corrects = 0
        for inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            # labels = labels.to(device)
                            
            outputs = model(inputs)
            _, y = torch.max(outputs, 1)
            
            y = y.cpu()
            y = convert_imagenet_to_cat_dog_label(y)
            running_corrects += np.sum(y == labels.numpy())

            # cat 0, dog 1
            # preds = torch.where(reduce(torch.bitwise_or, [preds==y for y in catIds]), 0, -1)
            # preds[reduce(torch.bitwise_or, [preds==y for y in dogIds])] = 1
            
            # running_corrects += np.sum(y == t.numpy())
        
        epoch_acc[phase] = 1.0 * running_corrects / len(dataloaders[phase].dataset)
    
    return epoch_acc['val'],epoch_acc['train']

def evaluate_onnx_fragment(onnxFragment, dataset_val):
    # fragmentFiles
    fullNet = Net([onnxFragment])
    macs, params = get_macs_params(fullNet[0])
    acc = accuracy_score_net(fullNet, dataset_val, bs=256)
    # print(acc, macs, params)
    return acc, macs, params

def get_accuracy_onnxfragments(onnxFragments, dataset_val):
    count = 0
    for x,t in tqdm(load_dl(dataset_val, batch_size=256, shuffle=False, num_workers=0)):
        outputs = execute_fragments(onnxFragments, x.numpy())
        y = outputs[-1]
        # y = model(x)
        # y = y.cpu()
        y = np.argmax(y, 1)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset_val)
    # print('accuracy', accuracy)
    return accuracy

from tqdm import tqdm
from stitchnet.stitchonnx.report import Report, ReportPlants, ReportKNNHFDataset
import traceback
import os

def run_experiment(result_name=None, nets=None, data_score=None, dataset_val=None, dataset_train=None, scoring_method='CKA',branching_factor=3, threshold=0.9, total_threshold=0.9, max_depth=16, eval_batch_size=16):
    k = 0
    if os.path.exists(f'./_results/{result_name}.txt'):
        with open(f'./_results/{result_name}.txt', 'r') as f:
            k = len(f.read().split('\n'))
    scoreMapper = ScoreMapper(nets, data_score, scoring_method=scoring_method)
    with ReportKNNHFDataset(eval_batch_size, f'./_results/{result_name}.txt', 'a') as report:
        # for _ in tqdm(range(50)):
        generator = generate_networks(nets, scoreMapper, data_score, 
                              threshold=threshold, totalThreshold=total_threshold, 
                              maxDepth=max_depth, sample=False, K=branching_factor)
        for i,(s,net) in enumerate(generator):
            try:
                netname = f"_results/{result_name}/net{k:03}"
                report.evaluate(nets, net, netname, s, dataset_val, dataset_train)
                net.save(netname)
                k += 1
            except Exception as e:
                print('[WARNING]', e)
                # traceback.print_exc()
                pass
            
def get_stitching_data_from_dataset(dset, batch_size=32, shuffle=False, seed=47):
    inps = []
    if shuffle:
        dset = dset.shuffle(seed=seed)
    
    if hasattr(dset,'take'):
        dset = dset.take(range(batch_size))
    else:
        dset = dset.select(range(batch_size))

    for x in tqdm(dset):
        try:
            inps += x['pixel_values']
        except:
            pass

    data_score = np.stack(inps)
    return data_score

from evaluate import evaluator
from stitchnet.stitchonnx.report import NetEvalPipeline

def evaluate_net(net, dataset_train, dataset_val, label_column='label'):
    task_evaluator = evaluator("image-classification")
    pipe = NetEvalPipeline(dataset_train, net, label=label_column)
    
    valresult = task_evaluator.compute(pipe, data=dataset_val, metric="accuracy", input_column='pixel_values', label_column=label_column)
    valacc = valresult['accuracy']
    trainresult = task_evaluator.compute(pipe, data=dataset_train, metric="accuracy", input_column='pixel_values', label_column=label_column)
    trainacc = trainresult['accuracy']
    
    return valacc, trainacc
