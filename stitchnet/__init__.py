__version__ = '0.2.0'

from glob import glob
from pathlib import Path
from skl2onnx.helpers.onnx_helper import load_onnx_model
import os
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import load_onnx_model
from graphviz import Digraph

from .stitchonnx.utils import generate_networks, ScoreMapper
from .stitchonnx.utils import Net
from .stitchonnx.utils import change_input_dim, get_fragments, Net
from .stitchonnx.utils import load_hf_train_val_dset_with_test_split
from .stitchonnx import utils

def set_device(device):
    utils.device = device
    
def get_cache_folder():
    cache_folder = os.getenv('SN_HOME', None)  # Check if SN_HOME environment variable is set

    if cache_folder is None:  # If SN_HOME is not set, use default cache directories based on the OS
        if os.name == 'posix':  # Linux or macOS
            cache_folder = os.path.expanduser('~/.cache/stitchnet/')
        elif os.name == 'nt':  # Windows
            cache_folder = os.path.expandvars('%LOCALAPPDATA%/stitchnet/')
        else:
            raise EnvironmentError("Unsupported operating system. Please set the SN_HOME environment variable.")

    return cache_folder

def generate(data_score, threshold=0.9, totalThreshold=0.5, maxDepth=10, sample=False, K=None):
    
    # get cache folder
    cache_folder =  get_cache_folder()
    os.makedirs(cache_folder, exist_ok=True)

    # download nets
    os.makedirs(os.path.join(cache_folder,"_models"), exist_ok=True)
    modelpaths = [
        ("alexnet", "_models/alexnet.onnx"),
        ("densenet121", "_models/densenet121.onnx"),
        ("mobilenet_v3_small", "_models/mobilenet_v3_small.onnx"),
        ("resnet50", "_models/resnet50.onnx"),
        ("vgg16", "_models/vgg16.onnx"),
    ]
    for modelpath in tqdm(modelpaths):
        model_name = modelpath[0]
        dest_path = os.path.join(cache_folder,modelpath[1])        
        if not os.path.exists(dest_path):
            model = torch.hub.load("pytorch/vision:v0.13.1", model_name, weights="IMAGENET1K_V1")
            model.eval()
            torch.onnx.export(model, torch.ones(1,3,224,224), dest_path, verbose=True)
    
    # generate fragments
    nets = []
    modelnames = sorted(glob(os.path.join(cache_folder,'_models/*.onnx')))
    print(modelnames)
    os.makedirs(os.path.join(cache_folder,f"_models/fragments"), exist_ok=True)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    for i,modelname in tqdm(enumerate(modelnames)):
        folder = os.path.join(cache_folder,f"_models/fragments/net{i:03}/")
        if not os.path.exists(folder):
            model_onnx1 = load_onnx_model(modelname)
            change_input_dim(model_onnx1)
            fragments1 = get_fragments(model_onnx1, x, 1)
            net = Net(fragments1, i)
            for j,fragment in enumerate(net):
                os.makedirs(folder, exist_ok = True)
                filename = folder+f'fragment{j:03}.onnx'
                if not os.path.exists(filename):
                    save_onnx_model(fragment.fragment, filename)
    
    # load fragments
    netsFiles = sorted(glob(os.path.join(cache_folder,'_models/fragments/net*')))
    nets = []
    for i,netsFile in tqdm(enumerate(netsFiles)):
        fragmentFiles = sorted(glob(str(Path(netsFile)/'fragment*.onnx')))
        onnxFragments = []
        for fragmentFile in fragmentFiles:
            onnxFragment = load_onnx_model(fragmentFile)
            onnxFragments.append(onnxFragment)
        net1 = Net(onnxFragments, i)
        nets.append(net1)

    scoreMapper = ScoreMapper(nets, data_score, scoring_method='CKA')

    generator = generate_networks(nets, scoreMapper, data_score, 
                          threshold=threshold, totalThreshold=totalThreshold, 
                          maxDepth=maxDepth, sample=sample, K=K)
    
    if sample:
        return next(generator)
    
    return generator

def load_hf_dataset(name, train_split='validation', val_split='test', label_column='labels', seed=47, streaming=False):
    return load_hf_train_val_dset_with_test_split(name, train=train_split, 
                                                        val=val_split, 
                                                        label=label_column,
                                                        seed=seed, streaming=streaming)