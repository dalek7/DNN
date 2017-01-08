# cuDNN
```bash
tar -xvf cudnn-8.0-linux-x64-v5.1
sudo cp cuda/include/*.h /usr/local/cuda/include
sudo cp cuda/lib64/*.so* /usr/local/cuda/lib64
```

# dependencies
```bash
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml
```

# caffe download
```bash
git clone https://github.com/BVLC/caffe.git
```

## Makefile.config
Checking the OpenCV version
```bash
pkg-config --modversion opencv
```

## caffe build (CMake)
```bash
mkdir build & cd build
cmake ..
make pycaffe
make all
make test
sudo make install
```
