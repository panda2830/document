# 人工智能环境配置-Ubuntu22.04

1. 安装CUDA
2. 安装CUDNN
3. 安装CMake
4. 安装opencv
5. 安装tensorrt(推理加速)
6. 安装anacoda(pytorch环境)
7. 安装labelimg(图片标注)
8. 下载yolov5项目(模型训练)
9. 下载tensorrtx项目(模型转换)

## 1.安装cuda

1. 查看显卡驱动信息

     `nvidia-smi`
     ![nvidia-smi](./images/nvidia-smi.png)
     CUDA Version显示本驱动可用的最高cuda版本

     下载对应cuda版本的cuda https://developer.nvidia.com/cuda-toolkit-archive

2. 运行cuda安装程序
     `sudo sh cuda_11.6.2_510.47.03_linux.run`

     选择Continue

     ![cuda-install](./images/cuda1.png)

     输入accept

     ![cuda-install](./images/cuda2.png)

     取消第一个选项，然后回车install

     ![cuda-install](./images/cuda3.png)

     安装成功会显示需要设置的环境变量

     ![cuda-install](./images/cuda4.png)


3. 设置环境变量

     输入命令
     `sudo gedit ~/.bashrc`

     添加下面内容到文件最后

     ```shell
     export PATH=$PATH:/usr/local/cuda-11.6/bin
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.6/lib64
     ```

     ![cuda-install](./images/cuda5.png)

     刷新环境变量

     `source ~/.bashrc`

4. 验证安装是否成功

     `nvcc -V`

     ![cuda-install](./images/cuda6.png)



## 2.安装cudnn

1. 下载cudnn https://developer.nvidia.com/rdp/cudnn-archive

2. 解压 xxx为下载的版本号

     `tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz`

3. 安装
     ```shell 
     sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
     sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
     sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
     ```

4. 验证安装

     `cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2`

     ![cudnn-install](./images/cudnn1.png)

## 3.安装cmake

1. 下载cmake3.14.7.tar.gz https://cmake.org/files/v3.14/

2. 解压

     `tar -zxvf cmake-3.14.7.tar.gz`

3. 进入cmake目录

     `cd cmake-3.14.7/`

4. 验证编译

     `./bootstrap`

     ![cmake-install](./images/cmake1.png)


5. 编译

     `make -j16`

6. 安装

     `sudo make install`

7. 验证安装

     `cmake --version`

     ![cmake-install](./images/cmake2.png)

## 4.opencv

1. 下载opencv4.7 https://opencv.org/releases/

     ![cmake-install](./images/opencv1.png)

2. 解压并进入opencv目录

     `unzip opencv-4.7.0.zip`

     `cd opencv-4.7.0/`

3. 安装依赖包

     `sudo apt-get update`

     ```shell
     sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
          libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
          libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
          gfortran openexr libatlas-base-dev python3-dev python3-numpy \
          libtbb2 libtbb-dev libdc1394-25
     ```
4. 创建目录并进入目录

     `mkdir build`

     `cd build`

5. cmake编译

     `sudo cmake -DCMAKE_BUILD_TYPE=Release -DOPENCV_GENERATE_PKGCONFIG=ON -DCMAKE_INSTALL_PREFIX=/usr/local .. `

     ![opencv-install](./images/opencv4.png)

6. 编译

     `sudo make -j16`

7. 安装

     `sudo make install `

     打开文件,添加 `/usr/local/lib`到文件中

     `sudo gedit /etc/ld.so.conf.d/opencv4.conf `

     `sudo ldconfig`

     ![opencv-install](./images/opencv2.png)

     安装updatedb命令

     `sudo apt-get install mlocate`

     `sudo updatedb`

8. 验证opencv安装

     `pkg-config --modversion opencv4`

     ![opencv-install](./images/opencv3.png)

## 5.tensorrt

1. 下载tensorrt https://developer.nvidia.com/nvidia-tensorrt-8x-download

     ![tensorrt-install](./images/tensorrt1.png)

2. 解压 

     `tar -zxvf TensorRT-8.5.3.1.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz`


3. 设置环境变量，`path/to`换成解压目录的上级目录

     `sudo gedit ~/.bashrc`

     ```shell
     export LD_LIBRARY_PATH=/path/to/TensorRT-8.5.3.1/lib:$LD_LIBRARY_PATH
     export LIBRARY_PATH=/path/to/TensorRT-8.5.3.1/lib::$LIBRARY_PATH
     ```

     ![tensorrt-install](./images/tensorrt2.png)

     刷新环境变量

     `source ~/.bashrc`


4. 验证安装

     `cd TensorRT-8.5.3.1/samples/sampleOnnxMNIST`

     `make -j16`

     在文件夹 `TensorRT-8.5.3.1/targets/x86_64-linux-gnu/bin`下会有生成的可执行文件sample_onnx_mnist

     进入文件夹并运行该可执行文件

     `./sample_onnx_mnist`

     显示下图则安装成功

     ![tensorrt-install](./images/tensorrt3.png)

## 6.anaconda

1. 下载anaconda3 https://www.anaconda.com/download

     ![anaconda-install](./images/anaconda1.png)

2. 安装

     `sh ./Anaconda3-2023.03-Linux-x86_64.sh`

     - 查看许可，输入回车
     - 空格，空格，空格...
     - 接受许可，输入yes      
     - 安装位置，默认即可，输入回车
     - 是否初始化终端？输入yes
     - 重启终端

3. 创建虚拟环境

     `conda create -n pytorch python=3.9`

     - 输入y确认安装

4. 进入虚拟环境

     `conda activate pytorch`

5. 安装pytorch，pytorch官网 https://pytorch.org

     ![anaconda-install](./images/pytorch1.png)

     `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

     - 输入y确认安装

## 7.安装labelImg

### 7.1pip换源

1. 家目录创建.pip目录

     `mkdir ~/.pip`

2. 创建pip配置文件
     
     `touch ~/.pip/pip.conf`

3. 打开pip配置文件

     `gedit ~/.pip/pip.conf`

4. 添加以下内容

     ```shell
     [global]
     timeout = 6000
     index-url = http://mirrors.aliyun.com/pypi/simple
     [install]
     trusted-host=mirrors.aliyun.com

     ```

### 7.2安装labelImg依赖包

1. 进入pytorch环境

     `conda activate pytorch`

2. 安装labelImg依赖包

     1. `pip install pyqt5`

     2. `pip install pyqt5-tools`

     3. `pip install lxml`

     4. `sudo apt install libxcb-xinerama0`

     5. `pip install labelimg`

### 7.3验证安装

1. labelImg注意**Img的I的大写**

     `labelImg`

## 8.yolov5项目






