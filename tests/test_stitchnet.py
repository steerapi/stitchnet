from stitchnet import __version__
from stitchnet import load_hf_dataset
from stitchnet import get_cache_folder, generate
import numpy as np
from tqdm import tqdm
from stitchnet.stitchonnx import utils
utils.device = 'cpu'
    
def test_version():
    assert __version__ == '0.2.0'

def test_load_hf_dataset():
    # load the beans dataset from huggingface
    dataset_train, dataset_val = load_hf_dataset('beans', train_split='validation', val_split='test', label_column='labels', seed=47)
    assert len(dataset_train) == 133
    
def test_generate():
    dataset_train, dataset_val = load_hf_dataset('beans', train_split='validation', val_split='test', label_column='labels', seed=47)
    stitching_dataset = np.vstack([x['pixel_values'] for x in tqdm(dataset_train.select(range(32)))])
    score,net = generate(stitching_dataset, threshold=0, totalThreshold=0, maxDepth=10, K=1, sample=True)    
    assert abs(score) > 0
    
def test_fit():
    dataset_train, dataset_val = load_hf_dataset('beans', train_split='validation', val_split='test', label_column='labels', seed=47)
    stitching_dataset = np.vstack([x['pixel_values'] for x in tqdm(dataset_train.select(range(32)))])
    score,net = generate(stitching_dataset, threshold=0, totalThreshold=0, maxDepth=10, K=1, sample=True)    
    net.fit(dataset_train, label_column="labels")
    assert net.knn is not None
    
def test_evaluate_dataset():
    
    dataset_train, dataset_val = load_hf_dataset('beans', train_split='validation', val_split='test', label_column='labels', seed=47)
    stitching_dataset = np.vstack([x['pixel_values'] for x in tqdm(dataset_train.select(range(32)))])
    score,net = generate(stitching_dataset, threshold=0, totalThreshold=0, maxDepth=10, K=1, sample=True)    
    net.fit(dataset_train, label_column="labels")
    result = net.evaluate_dataset(dataset_val, label_column='labels')
    
    assert abs(result['accuracy']) > 0