StitchNet
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

