StitchNet: Composing Neural Networks from Pre-Trained Fragments
=============


Installation
=============

    pip install stitchnet
    
Usage
=============
    
    import stitchnet
    
    # prepare stitching data D
    from stitchnet import load_hf_dataset
    # load the beans dataset from huggingface
    dataset_train, dataset_val = load_hf_dataset('beans', train_split='validation', val_split='test', label_column='labels', seed=47)
    
    # generate stitchnets
    import numpy as np
    from tqdm import tqdm
    stitching_dataset = np.vstack([x['pixel_values'] for x in tqdm(dataset_train.select(range(32)))])
    score,net = generate(stitching_dataset, threshold=0.8, totalThreshold=0, maxDepth=10, K=2, sample=True)
        
    # save onnx
    net.save_onnx('./_data/net')
        
    # draw the stitchnet
    net.draw_svg('./_data/net')
    
    # train a classifier
    net.fit(dataset_train, label_column="labels")
    
    # use it for prediction
    net.predict_files(['./_results/test.jpg'])
    # [{'score': [0.8, 0.2, 0.0], 'label': 0}]
    
    # evaluate the classifier
    net.evaluate_dataset(dataset_val, label_column='labels')
    # {'accuracy': 0.7421875}

CUDA
=============
See https://pytorch.org/get-started/previous-versions/ to install appropriate version. For example

    # CUDA 11.6
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116


Experiment Notebooks
=============

1. Download dogs and cats dataset from https://www.kaggle.com/c/dogs-vs-cats/data and put train data in _data/dogs_cats/raw/train folder
2. See 00_prepare_data.ipynb to split the images into cats and dogs folder
3. See 01_download_networks.ipynb to download the pretrained networks from Torchvision
4. See 02_generate_fragments.ipynb to generate fragments from the pretrained networks
5. See 03_stitchnet.ipynb to generate stitchnets
6. See 04_render_graph.ipynb to create svg images of the network graphs using netron
7. See 05_eval_original_networks.ipynb for evaluating the original pretrained networks
8. See 06_finetuning.ipynb to generate the finetuning result
9. See 07_ensemble.ipynb to generate the ensemble result
10. See 08_number_of_samples_for_stitching.ipynb for experimenting with varying number of samples to use when stitching
11. See 09_plot_results.ipynb plot figures of the results for the paper


Installation using conda
=============

Create a new conda env

    conda create -n stitchnet python=3.10
    
Activate stitchnet conda env

    conda activate stitchnet

For conda and NVIDIA gpu, please also install for CUDA runtime on onnx

    conda install -c conda-forge cudnn
    
Install poetry

    curl -sSL https://install.python-poetry.org | python3 -

Install dependencies using poetry 

    poetry install

