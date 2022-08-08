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

# PROVIDERS = ['CPUExecutionProvider']
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

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
    print('[POTENTIAL SPLIT LOCATIONS]:', inputs)
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
            print('[WARNING]:', i, inp, e)
            # traceback.print_exc()
            # pass
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
    outputs = []
    for f in fragments:
        change_input_dim(f)
        # print('execute_fragments', 'change input', f.graph.input)
        ort_sess = ort.InferenceSession(f.SerializeToString(), providers=PROVIDERS)
        inputs = {}
        # print("exec", "input", f.graph.input[0].name, f.graph.input[0].type)
        inputs[f.graph.input[0].name] = x
        o = ort_sess.run(None, inputs)
        x = o[0]
        outputs.append(x)
    return outputs

from collections import defaultdict
import hashlib
# print(hashlib.md5("whatever your string is".encode('utf-8')).hexdigest())
def hash_model(f):
    change_input_dim(f)
    hashf = hashlib.md5(f.SerializeToString()).hexdigest()
    return hashf

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
    
    def get_id(self):
        return self.id
    def get_scores(self, x1, data):
        tX = torch.from_numpy(x1)
        score_fragments = []
        for i,f in enumerate(self.fragments[:-1]):
            x2 = self.get_output(f,data)
            tY = torch.from_numpy(x2)
            score = get_score(tX, tY)
            score_fragments.append((score, self.fragmentCs[i+1]))
        return score_fragments
    
    def evaluate(self, x):
        hashx = hashlib.md5(x.data.tobytes()).hexdigest()
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
        hashx = hashlib.md5(x.data.tobytes()).hexdigest()
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
    
    
def sample_index(X,Y):
    num_samples = min(min(X.shape[1],Y.shape[1])*10, X.shape[0], Y.shape[0])
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
    
    A = acts2.T @ acts1.T.pinverse()
    
    tw = torch.from_numpy(w)
    nw = torch.einsum('ij, jk -> ik', tw, A)
    nw = nw.numpy()
    return nw
    
def adjust_w_conv(tX, tY, w):
    if tY.shape[-2]!=tX.shape[-2] and tY.shape[-1]!=tX.shape[-1]:
        up = torch.nn.UpsamplingBilinear2d((tY.shape[-2],tY.shape[-1]))
        tX = up(tX)
        # addLayers += [up]

    acts1 = tX.permute((0,2,3,1)).reshape(-1,tX.shape[1])
    acts2 = tY.permute((0,2,3,1)).reshape(-1,tY.shape[1])

    indexX = sample_index(acts1, acts2)
    acts1 = acts1[indexX,:]
    acts2 = acts2[indexX,:]

    A = acts2.T @ acts1.T.pinverse()
    
    tw = torch.from_numpy(w)
    nw = torch.einsum('ijkl, jn -> inkl', tw, A)
    nw = nw.numpy()
    return nw

def adjust_w(tX, tY, w):
    if w.ndim == 2:
        return adjust_w_linear(tX, tY, w)
    else:
        return adjust_w_conv(tX, tY, w)

# def clone_node(node):
#     newnode = helper.make_node(
#         node.op_type,                  # name
#         node.input, # inputs
#         node.output,                  # outputs
#         mode='constant',        # attributes
#     )
#     return newnode

def get_score_fragments(fragment1: Fragment, fragment2: Fragment, data):
    x1 = fragment1.get_output(data)
    x2 = fragment2.get_input(data)
    tX = torch.from_numpy(x1)
    tY = torch.from_numpy(x2)
    score = get_score(tX, tY)
    return score

def stitch_fragments(fragment1: Fragment, fragment2: Fragment, data):
    # list_ops(fragment1.fragment)
    # fragment1 = copy.deepcopy(fragment1)
    # fragment2 = copy.deepcopy(fragment2)
    
    try:
        x1 = fragment1.get_output(data)
        x2 = fragment2.get_input(data)
    except Exception as e:
        traceback.print_exc()
        print('-------fragment1 ops-------')
        list_ops(fragment1.fragment)
        print('-------fragment2 ops-------')
        list_ops(fragment2.fragment)
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
        print('len(inpnodes)', len(inpnodes))
        print('newFragment', newFragment, fragment2.fragment)
        list_ops('newFragment', newFragment)
        list_ops('fragment2', fragment2.fragment)
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
        newnet = onnxoptimizer.optimize(newnet)
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
        print()

def get_all_fragments(nets):
    fragments = [f for net in nets for f in net]
    return fragments

class ScoreMapper:
    def __init__(self, nets, data):
        self.nets = nets
        self.scoreMap = {}
        self.data = data
    def score(self, x1, hashId=None):
        hashx = hashlib.md5(x1.data.tobytes()).hexdigest()
        # if hashId is None:
        # else:
        #     hashx = hashId
        # # print('hashx in self.scoreMap', hashx in self.scoreMap)
        # if hashx in self.scoreMap:
        #     return self.scoreMap[hashx]
        scores = []
        for net2 in tqdm(self.nets):
             scores += net2.get_scores(x1,self.data)
        scores = sorted(scores, key=lambda x:x[0], reverse=True)
        # disabled caching for now since it is taking up too much memory
        self.scoreMap[hashx] = scores
        return self.scoreMap[hashx]
        # return scores

def find_next_fragment(curr, scoreMapper, data, threshold=0.5, sample=False):
    x1 = curr.get_output(data)
    scores = scoreMapper.score(x1)
    print('potential next fragments:', len(scores))
    scores = [score for score in scores if score[0]>threshold]
    if sample:
        scores = [np.random.choice(scores)]
    print(f'potential next fragments after thresholding of {threshold}:', len(scores))
    for s,nextf in scores:
        score = get_score_fragments(curr, nextf, data)
        if score < threshold:
            continue
        yield score,nextf
        
def get_net_id(curr, nextf):
    fId = curr.get_id()
    if type(fId[0]) is tuple:
        nId = fId[0] + (nextf.get_id(),)
    else:
        nId = (fId,)
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

def recursive_stitching(curr, scoreMapper, data, threshold=0.9, totalThreshold=0.5, totalscore=1, sample=False):
    for score,nextf in find_next_fragment(curr, scoreMapper, data, threshold, sample):
        totalscore_nextf = totalscore*score
        if totalscore_nextf < totalThreshold:
            continue
        print('totalscore', totalscore_nextf);
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
                for _score, _curr  in recursive_stitching(newcurr, scoreMapper, data, threshold, totalThreshold, totalscore_nextf, sample):
                    yield _score, _curr
        except Exception as e:
            # catch death end path with errors
            print('ERROR', e)
            traceback.print_exc()
            # raise e
            # pass
            
def generate_networks(nets, scoreMapper, data, threshold=0.9, totalThreshold=0.5, sample=False):
    fragments = [f for net in nets for f in net]
    starts = [f for f in fragments if f.fragment.graph.name == 'start']
    ends = [f for f in fragments if f.fragment.graph.name == 'end']
    middles = [f for f in fragments if f.fragment.graph.name not in ['start','end']]
    for start in starts:
        for score, curr in recursive_stitching(start, scoreMapper, data, threshold, totalThreshold, sample):
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
    for fragmentC in net:
        change_input_dim(fragmentC.fragment)
        ort_sess1 = ort.InferenceSession(fragmentC.fragment.SerializeToString(), providers=PROVIDERS)
        inputs = {}
        inputs[fragmentC.fragment.graph.input[0].name] = data
        outputs = ort_sess1.run(None, inputs)
        data = outputs[0]
        # print(data.shape)
    return data
    
from .ptCKA import linear_CKA
@torch.no_grad()
def get_score(X, Y):
    # X = torch.from_numpy(X)
    # Y = torch.from_numpy(Y)
    # print('X1', X.shape)
    # print('Y1', Y.shape)
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
        num_samples = min(min(X.shape[1],Y.shape[1])*10, X.shape[0], Y.shape[0])
        # print(num_samples, X.shape, Y.shape)
        p = torch.ones(X.shape[0])
        indexX = p.multinomial(num_samples=num_samples)
        # p = torch.ones(Y.shape[0])
        # indexY = p.multinomial(num_samples=num_samples)
        # print(X[indexX,:].shape, Y[indexY,:].shape)
        s = linear_CKA(X[indexX,:],Y[indexX,:]).cpu().item()
    elif X.ndim == 4 and Y.ndim == 2:
        pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        flat = torch.nn.Flatten()
        X = flat(pool(X))
        b = X.shape[0]
        X = X.reshape(b,-1)
        Y = Y.reshape(b,-1)
        num_samples = min(1000,X.shape[1],Y.shape[1])
        p = torch.ones(X.shape[1])
        indexX = p.multinomial(num_samples=num_samples)
        p = torch.ones(Y.shape[1])
        indexY = p.multinomial(num_samples=num_samples)
        X = X[:,indexX]
        Y = Y[:,indexY]
        s = linear_CKA(X,Y).cpu().item()
    else:
        # TODO: check b > 2*min(X.shape[1],Y.shape[1])
        b = X.shape[0]
        X = X.reshape(b,-1)
        Y = Y.reshape(b,-1)
        num_samples = min(1000,X.shape[1],Y.shape[1])
        p = torch.ones(X.shape[1])
        indexX = p.multinomial(num_samples=num_samples)
        p = torch.ones(Y.shape[1])
        indexY = p.multinomial(num_samples=num_samples)
        X = X[:,indexX]
        Y = Y[:,indexY]
        s = linear_CKA(X,Y).cpu().item()
        
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

# @torch.no_grad()
def accuracy_score_model(model, dataset, bs=64):
    count = 0
    for x,t in tqdm(torch.utils.data.DataLoader(dataset, batch_size=bs)):
        y = model(x)
        y = np.argmax(y.detach().numpy(), 1)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy
    
def accuracy_score_net(net, dataset, bs=64):
    count = 0
    for x,t in tqdm(torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)):
        y = evalulate_stitchnet(net, x)
        # print('len(ys)', len(ys))
        # y = ys[0]
        # print('y.shape', y.shape)
        y = np.argmax(y, 1)
        y = convert_imagenet_to_cat_dog_label(y)
        count += np.sum(y == t.numpy())
    accuracy = 1.*count/len(dataset)
    return accuracy

# data cat and dog
# https://www.kaggle.com/competitions/dogs-vs-cats/rules

import torchvision
import torch
from torchvision.models import ResNet50_Weights
import os
from pathlib import Path

def load_cats_and_dogs_dset(folder="train"):
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    dataset = torchvision.datasets.ImageFolder(f'_data/dogs_cats/{folder}', transform=preprocess)
    return dataset

def load_dl(dset, bs):
    dl = torch.utils.data.DataLoader(dset, shuffle=True, batch_size=bs)
    return dl

# def load_cats_and_dogs():
#     weights = ResNet50_Weights.IMAGENET1K_V1
#     preprocess = weights.transforms()
#     dataset = torchvision.datasets.ImageFolder(Path(os.path.dirname(__file__)+'/../_data/dogs_cats/train').resolve(), transform=preprocess)
#     dl = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64)
#     return dl


def find_network(nets, data, maxround=10):
    fragments = [f for net in nets for f in net]
    starts = [f for f in fragments if f.fragment.graph.name == 'start']
    ends = [f for f in fragments if f.fragment.graph.name == 'end']
    middles = [f for f in fragments if f.fragment.graph.name not in ['start','end']]
    
    start = np.random.choice(starts)
    curr = start
    totalscore = 1
    while maxround >= 0:
        nextf = np.random.choice(middles + ends)
        score = get_score_fragments(curr, nextf, data)
        if score < threshold:
            continue

        # print('pick')
        # list_ops(nextf.fragment)
        # if get_score_fragments(curr, nextf, data) < 0.1:
        #     continue
        try:
            curr_fragment = stitch_fragments(curr, nextf, data)
            curr_net = Net([curr_fragment])
            curr = curr_net[0]
            totalscore *= score
            # print('c', curr)
            if nextf.fragment.graph.name == 'end':
                break
        except:
            # traceback.print_exc()
            pass
        maxround-=1
        
    if nextf.fragment.graph.name != 'end':
        nextf = np.random.choice(ends)
        # print('pickend')
        # list_ops(nextf.fragment)
        curr_fragment = stitch_fragments(curr, nextf, data)
        curr_net = Net([curr_fragment])
        score = get_score_fragments(curr, nextf, data)
        totalscore *= score
    return curr_net, totalscore
        