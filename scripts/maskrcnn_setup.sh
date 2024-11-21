#! /bin/bash
python -m pip install --upgrade pip
pip install ninja yacs cython matplotlib tqdm opencv-python

INSTALL_DIR=$PWD

cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install

cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
python setup.py build develop

cd $INSTALL_DIR
rm -r cocoapi/cityscapesScripts/apex/maskrcnn-benchmark/

unset INSTALL_DIR

printf "\n maskrcnn_benchmark is all set\n\n"
