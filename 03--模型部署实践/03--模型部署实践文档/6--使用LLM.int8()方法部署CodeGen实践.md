# 使用LLM.int8()方法部署CodeGen实践



## <span style='color:brown'>内网部署的具体操作指南</span>

1. docker image

   - gameai/torchserver:0.1
   - -v   /workspace/codegen-6B

2. 安装依赖包

   1. transformers

      1. 先卸载原镜像中的`transformers`版本；

      2. 在安装内网源中的 `transformers`：

         ```shell
         $ pip install transformers -i https://apt.2980.com/pypi/simple
         ```

      3. 安装 `transformers-main` 官网安装包

         ```shell
         $ cd transformers-main
         $ python setup.py install
         ```

   2. accelerate

      ```shell
      $ pip install accelerate -i https://apt.2980.com/pypi/simple
      ```

   3. bitsandbytes

      ```shell
      $ cd bitsandbytes-main
      $ python setup.py install
      ```

3. Torch版本更换

   1. 卸载原镜像中的Torch版本

      ```shell
      $ pip uninstall torch
      ```

      备注：此处 `torch 1.10.0a0+0aef44c`

   2. 安装新版本的Torch

      ```shell
      $ pip install torch==1.13.0 -i https://apt.2980.com/pypi/simple
      ```

4. 将模型压缩所需的 `bitsandbytes` 相关 `so` 文件复制到容器中对应位置

   ```shell
   $ cp -a /workspace/codegen-6B/bitsandbytes/bitsandbytes/.  /opt/conda/lib/python3.8/site-packages/bitsandbytes-0.35.4-py3.8.egg/bitsandbytes/
   ```

5. 返回 `notebook`，验证代码的可运行性

   即可。

<span style='color:brown'>**预留问题：**</span>

- "Setting 'pad_token_id' to 'eos_token_id': 50256 for open-end generation".

- Solution

  ```python
  model.generate(..., pad_tokne_id=50256, ...)
  ```



## <span style='color:brown'>Colab 上测试基于LLM.INT8()量化方法的实践</span>

问题描述：

- 在Colab上使用bitsandbytes库进行LLM.INT8()量化模型，出现报错：

  > 1、Required library version not found: libbitsandbytes_cuda112.so

问题解析：

- 基于Github的bitsandbytes源代码进行安装，会导致HuggingFace的INT8()量化方法无法有效使用，版本记录：0.35.4；

  ```shell
  $ pip install --quiet github+https://github.com/TimDettmers/bitsandbytes.git
  ```

解决方法：

- 更换版本库版本（回退版本==0.31.7）

  ```shell
  $ pip install -i https://test.pypi.org/simple/bitsandbytes
  ```

  将之前复制出来的编译文件放到 python/dist-package的指定地址。



## 资料集

### 1、**Anaconda Packages**

- repo.anaconda.com/pkgs/
- 内网地址：http://mvn-apt.2980.com/repository/anaconda-pkgs/

**Anaconda (pkgs/main channel)**

Added Sept 26, 2017 with the release of Anaconda 5.0 and conda 4.3.27, and includes packages built by Anaconda, Inc. with the new compiler stack. The majority of all new Anaconda, Inc. package builds are hosted here. Included in conda's defaults channel as the top priority channel. Of interest: https://www.anaconda.com/blog/developer-blog/utilizing-the-new-compilers-in-anaconda-distribution-5/

- Linux x86_64 (64-bit) [main/linux-64](https://repo.anaconda.com/pkgs/main/linux-64/) (major support)
- Linux x86 (32-bit) [main/linux-32](https://repo.anaconda.com/pkgs/main/linux-32/)
- Linux on Graviton2/ARM64 (64-bit) [main/linux-aarch64](https://repo.anaconda.com/pkgs/main/linux-aarch64/) (major support)
- Linux on IBM Z & LinuxONE (64-bit) [main/linux-s390x](https://repo.anaconda.com/pkgs/main/linux-s390x/) (major support)
- Linux ppc64le (64-bit) [main/linux-ppc64le](https://repo.anaconda.com/pkgs/main/linux-ppc64le/) (major support)
- MacOSX x86_64 (64-bit) [main/osx-64](https://repo.anaconda.com/pkgs/main/osx-64/) (major support)
- macOS Apple M1/ARM64 (64-bit) [main/osx-arm64](https://repo.anaconda.com/pkgs/main/osx-arm64/) (major support)
- Windows x86_64 (64-bit) [main/win-64](https://repo.anaconda.com/pkgs/main/win-64/) (major support)
- Windows x86 (32-bit) [main/win-32](https://repo.anaconda.com/pkgs/main/win-32/) (major support)
- platform agnostic (noarch) [main/noarch](https://repo.anaconda.com/pkgs/main/noarch/)



### 2、**Anaconda Documentation TroubleShooting**

- [Web-Link](https://docs.anaconda.com/navigator/troubleshooting/)

**TroubleShooting: **

- Nvigator error on start up
- Issues launching or initializing 
- PermissionError on macOS Catalina
- Access denied error



### 3、**CUDA 11 功能介绍**

CUDA 11使你能够利用新的硬件功能来加速HPC、基因组学、5G、渲染、深度学习、数据分析、数据科学、机器人技术以及更多不同的工作负载。

CUDA 11具有丰富的功能--从平台系统软件到开始开发GPU加速应用程序所需的一切。这篇文章概述了此版本中的主要软件功能：

- 支持NVIDIA Ampere GPU架构，包括新的NVIDIA A100 GPU，用于加速AI和HPC数据中心的纵向扩展和横向扩展；具有NVSwitch结构的多GPU系统，例如DGX A100和HGX A100；
- 多实例GPU(MIG)分区功能特别有利于云服务供应商(CSP)提高GPU利用率；
- 新的第三代Tensor Core可加速不同数据类型(包括TF32和Bfloat16)的混合精度矩阵运算；
- 用于任务图、异步数据移动、细粒度同步和L2缓存驻留控制的编程和API；
- CUDA库中线性代数、FFT和矩阵乘法的性能优化；
- 更新了用于跟踪、分析和调试CUDA应用程序的Nsight产品系列工具；
- 全面支持所有主要CPU架构，跨x86_64、Arm64服务器和POWER架构



**第三代多精度张量核**

NVIDIA A100中每个SM的四个大型张量核心(总共432个张量核心)为所有数据类型提供更快的矩阵乘法累加(MMA)运算：二进制、INT4、INT8、FP16、Bfloat16、FT32和FP64。

CUDA 11 添加了对新输入数据类型格式的支持：Bfloat16、TF32 和 FP64。Bfloat16 是另一种 FP16 格式，但精度降低，与 FP32 数值范围相匹配。它的使用导致更低的带宽和存储要求，因此更高的吞吐量。Bfloat16 通过 WMMA 在 cuda_bf16.h 中作为新的 CUDA C++`__nv_bfloat16`数据类型公开，并受到各种 CUDA 数学库的支持。 

TF32 是一种特殊的浮点格式，旨在与 Tensor Cores 一起使用。TF32 包括一个 8 位指数（与 FP32 相同）、10 位尾数（与 FP16 相同精度）和一个符号位。这是默认的数学模式，允许您在 DL 训练中获得超过 FP32 的加速，而无需对模型进行任何更改。最后，A100 为 MMA 操作带来了双精度 (FP64) 支持，WMMA 接口也支持双精度 (FP64)。



## 问题集

1、**AttributeError:  /opt/conda/bin/python:  undefined symbol: cudaRuntimeGetVersion**

相关参考链接：

- [cudaRuntimeGetVersion #1207](https://github.com/huggingface/diffusers/issues/1207)
- [add "conda install cudatoolkit" to dreambooth "training on 16GB" exapmle #1229](https://github.com/huggingface/diffusers/pull/1229)
- [undefined symbol: cudaRuntimeGetVersion Error #85](https://github.com/TimDettmers/bitsandbytes/issues/85)
- [TimDettmers / bitsandbytes](https://github.com/TimDettmers/bitsandbytes#requirements--installation)
- [Required library version not found: libbitsandbytes_cuda100.so](https://github.com/TimDettmers/bitsandbytes/issues/82)



> [Oxdevalias:](https://github.com/huggingface/diffusers/issues/1207#issuecomment-1308326232)
>
> Was able to workaround this with `setenv LD_LIBRARY_PATH    /usr/lib/x86_64-linux-gnu` which is where my libcudart.so is (Ubuntu 22.04). Not sure why it misses that.
>
> I had this error and was stumped by it until i realised that my CUDA toolkit had been installed ~incorrectly and was in a place where `bitsandbytes` couldn't find it. So when i installed PyTorch and the toolkit via `conda` instead of `pip`, the library was able to locate the toolkit and this error went away.
>
> But basically it seemed like to use `bitsandbytes` for the 8bit adam optimiser, i also needed ` conda install cudatoolkit` as well, which wasn't mentioned [here](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#training-on-a-16gb-gpu).
>
> When I tried to look up how to install `cudatoolkit`, `conda` is mentioned as an explicit method [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda):
>
> 1. Installation
>
>    To perform a basic install of all CUDA Toolkit components using Conda, 
>
>    ```shell
>    $ conda install cuda -C nvidia
>    ```
>
> 2. Uninstallation
>
>    ```shell
>    $ conda remove cuda
>    ```
>
> 3. Installing Previous CUDA Releases
>
>    All Conda packages released under a specific CUDA version are labeled with that release version. To install a previous version, include that label in the install command such as:
>
>    ```shell
>    $ conda install cuda -c nvidia/label/cuda-11.3.0
>    ```
>
>    
>
> 

备注：

>  **CUDA Toolkit**
>
> 该工具包包括GPU加速库、调试和优化工具、C/C++编译器以及用于部署应用程序的运行时库。
>
> NVIDIA Ampere 架构的支持包括下一代Tensor核心、混合精度模式、多实例GPU、高级内存管理和标准C++/Fortran并行语言结构。



### <span style='color:brown'>**Required library version not found: libbitsandbytes_cuda100.so** </span>

>  **[Titus-von-Koeller](https://github.com/Titus-von-Koeller)** commented [5 days ago](https://github.com/TimDettmers/bitsandbytes/issues/82#issuecomment-1326479324)
>
> Hey all, Tim is busy writing a new paper and I am completely swamped at work. I should be able to look into this in 7-14days from now. We're doing this in our free time. Thanks for your patience, Titus

##### <span style='color:blue'>**内网部署的解决方案**</span>

1. 借用外网GPU机器进行验证，然后将容器镜像化传入内网使用；
2. 将Colab env -->  clone env，然后在内网安装使用(不太可行)；
3. 将Colab运行的bitsandbytes安装包，直接打包放入内网使用( <span style='color:brown'>**验证可行**</span> )；



Colab确实文件地址:

```shell
$ pip install -i https://test.pypi.org/simple/ bitsandbytes
```

```shell
$ find / -name "libbitsandbytes_cuda114.so"
```

- `/usr/local/lib/python3.7/dist-packages/bitsandbytes/libbitsandbytes_cuda114.so`

打包 `bitsandbytes` 的包：

```shell
$ cp -R /usr/local/lib/python3.7/dist-packages/bitsandbytes  /content/drive/MyDrive
```

放入内网的地址：

```shell
$ cp -a /workspace/codegen-6B/bitsandbytes/bitsandbytes/.  /opt/conda/lib/python3.8/site-packages/bitsandbytes-0.35.4-py3.8.egg/bitsandbytes/
```

测试代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "model_name"
model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

text = "# quick sort"

output_sequances = model_8bit.generate(**tokenizer(text, return_tensors="pt")
                                       , max_new_tokens=64)
predict = tokenizer.decode(output_sequances[0], skip_special_tokens=True)
```

2、**how to create a `.condarc` file for Anaconda ?**

- [StackOverflow](https://stackoverflow.com/questions/29896309/how-to-create-a-condarc-file-for-anaconda)
- Using the `.condarc` conda configuration file

```shell
$ find / -name file_name

$ vim /opt/conda/.condarc
```



3、**Could not load dynamic library "libcudart.so.11.0"**

- [StackOverflow](https://stackoverflow.com/questions/70967651/could-not-load-dynamic-library-libcudart-so-11-0)
- [Github--issue](https://github.com/tensorflow/tensorflow/issues/45930)

解决方案：

1. find out where the "libcudart.so.11.0"

   ```shell
   $ sudo find / -name "libcudart.so.11.0"
   ```

   If the result shows nothing, please make sure you have install cuda or other staff that must install in your system.

2. add the path to environment file.

   ```shell
   # edit /etc/profile
   sudo vim /etc/profile
   # append path to "LD_LIBRARY_PATH" in profile file
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.1/targets/x86_64-linux/lib
   # make environment file work
   source /etc/profile
   ```

3. install

   ```shell
   $ conda install cudatoolkit
   ```

   

4、**How to change CUDA version**

- [StackOverflow](https://stackoverflow.com/questions/45477133/how-to-change-cuda-version)

解决方案：

1. Change the CUDA soft link to point on your desired CUDA version.

   ```shell
   # remove the old symlink
   $ sudo rm /usr/local/cuda
   $ ls /usr/local/cuda
   lrwxrwxrwx. 1 root root 20 Sep 14 08:03 /usr/local/cuda -> /usr/local/cuda-10.2
   $ sudo ln -sfT /usr/local/cuda/cuda-11.1  /usr/local/cuda
   $ ls /usr/local/cuda
   lrwxrwxrwx. 1 root root 26 Sep 14 13:25 /usr/local/cuda -> /usr/local/cuda/cuda-11.1/
   ```

   

5、**What is the difference between env, setenv, export and when to use ?**

- [StackOverflow](https://unix.stackexchange.com/questions/368944/what-is-the-difference-between-env-setenv-export-and-when-to-use)

问题描述：

- Recently i noticed we have 3 options to set environment variables:
  1. `export envVar1=1`
  2. `setenv envVar2=2`
  3. `env envVar3=3`

解决方案：

`export VARIABLE_NAME-'some value'`是在任何符合POSIX标准的shell(sh、dash、bash、ksh、zsh等)中设置环境变量的方法。如果该变量已经有了一个值，你可以使用`export VARIABLE_NAME`使其成为一个环境变量而不改变其值。

`setenv VARIABLE_NAME='some value'` 是设置环境变量的csh语法。`setenv` 在sh中不存在，csh 在脚本中极少使用，并且再过去20年的交互使用中被 `bash` 超越，所以可以忘记它，除非你遇到它。

除了在shebang行中，env命令很少有用。在没有参数的情况下，它会显示环境，但`export`阔的更好(排序，且经常被引用，以区分值中的换行和分隔值的换行)。当带参数调用时，它运行一个带有额外环境变量的命令，但同样的命令在没有 `env` 的情况下也可以运行(VAR= value mycommand 运行 mycommand, VAR设置为value，就像 `env VAR=value mycommand`)。`env` 在shebang行中有用的原因是：它执行PATH查找，而当用命令名调用时，它刚好不做其他事情。`env` 命令在运行一个只有几个环境变量的命令时非常有用，可以用`-i` ，或者不带参数来显示环境，包括 shell 没有导入的无效名称的变量。



6、**Removing NVIDIA CUDA Toolkit and installing new one**

- [AskUbuntu](https://askubuntu.com/questions/530043/removing-nvidia-cuda-toolkit-and-installing-new-one)

解决方案：

- Uninstall just nvidia-cuda-toolkit

  ```shell
  $ sudo apt-get remove nvidia-cuda-toolkit
  ```

- Uninstall nvidia-cuda-toolkit and it's dependencies

  ```shell
  $ sudo apt-get remove --auto-remove nvidia-cuda-toolkit
  ```

- Purging config/data

  ```shell
  $ sudo apt-get purge nvidia0cuda-toolkit
  ```

  or:

  ```shell
  $ sudo apt-get purge --auto-remove nvidia-cuda-toolkit
  ```

  Additionally, delete the `/opt/cuda` and `~/NVIDIA_GPU_Computing_SDK` folders if they are present. and remove the `export PATH=$PATH:/opt/cuda/bin` and `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/lib:/opt/cuda/lib64` lines of the `~/.bash_profile` file.  







