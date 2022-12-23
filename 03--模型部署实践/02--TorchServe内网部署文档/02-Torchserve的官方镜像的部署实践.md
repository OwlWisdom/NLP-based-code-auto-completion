# Torchserve的官方镜像的部署实践



**官方镜像：**

- [pytorch/torchserve](https://hub.docker.com/r/pytorch/torchserve/tags)

  - gpu

    ```shell
    docker pull pytorch/torchserve:0.5.3-gpu
    ```

  - cpu

    ```
    docker pull pytorch/torchserve:latest
    ```

  - 内网拉去dockerhub官方镜像的网址设置：

    ```shell
    docker pull hub.2980.com/dockerhub/pytorch/torchserve:0.5.3-gpu
    ```




## **使用方法：**

- 官方说明：[serve/docker/README](https://github.com/pytorch/serve/blob/master/docker/README.md)

### <span style='color:brown'>Start GPU container</span>

1. For GPU latest image with gpu devices 1 and 2:

   ```shell
   docker run --rm -it --gpus '"device=1,2"' -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:latest-gpu
   ```

2. For specific versions you can pass in the specific tag to use (ex: `0.1.1-cuda10.1-cudnn7-runtime`):

   ```shell
   docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:0.1.1-cuda10.1-cudnn7-runtime
   ```

3. For the latest version, you can use the `latest-gpu` tag:

   ```shell
   docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 pytorch/torchserve:latest-gpu
   ```

**Accessing TorchServe APIs inside container**

TorchServe 的推理和管理 API 可以分别通过 8080 和 8081 端口在 localhost 上访问。例子 ：

```shell
curl http://localhost:8080/ping
```



### <span style='color:brown'>Create torch-model-archiver from container</span>

要创建用于TorchServe部署的mar [model archive]文件，你可以使用以下步骤:

1. 通过分享你的本地model-store/any目录，包含自定义/example mar内容以及model-store目录（如果不在那里，请创建它）来启动容器：

   ```shell
   docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples pytorch/torchserve:latest
   ```

   > 如果使用Intel® Extension for PyTorch*启动容器，请在config.properties中添加以下几行，以启用IPEX和启动器的默认配置。
   >
   > ```properties
   > ipex_enable=true
   > cpu_launcher_enable=true
   > ```
   >
   > ```shell
   > docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/config.properties:/home/model-server/config.properties -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd)/examples:/home/model-server/examples torchserve-ipex:1.0
   > ```
   >
   > 

2. 如果你知道容器名称，请列出你的容器或跳过此内容

   ```shell
   $ docker ps
   ```

3. 绑定并获取运行容器的bash提示

   ```shell
   $ docker exec -it <container_name> /bin/bash
   ```

4. 如果你还没有这样做，请下载模型的权重（它们不是 repo 的一部分）

   ```shell
   curl -o /home/model-server/examples/image_classifier/densenet161-8d451a50.pth https://download.pytorch.org/models/densenet161-8d451a50.pth
   ```

5. 现在执行 torch-model-archiver 命令，例如：

   ```shell
   torch-model-archiver --model-name densenet161 --version 1.0 --model-file /home/model-server/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/examples/image_classifier/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /home/model-server/examples/image_classifier/index_to_name.json --handler image_classifier
   ```

6. desnet161.mar 文件应该存在于 /home/model-server/model-store



### <span style='color:brown'>Running TorchServe in a Production Docker Environment</span>

在用Docker在生产中部署torchserve时，你可能要考虑以下方面/docker选项。

- shared memory size

  - `shm-size` - shm-size参数允许你指定一个容器可以使用的共享内存。它使内存密集型的容器通过对分配的内存进行更多的访问而运行得更快。

- user limits for system resources

  - `--ulimit  memlock=-1` : Maximum locked-in-memory address space
  - `--ulimit  stack` : Linux stack size

  The current ulimit values can be viewed by executing `ulimit -a`. A more exhaustive set of options for resource constraining can be found in the Docker Documentation [here](https://docs.docker.com/config/containers/resource_constraints/), [here](https://docs.docker.com/engine/reference/commandline/run/#set-ulimits-in-container---ulimit) and [here](https://docs.docker.com/engine/reference/run/#runtime-constraints-on-resources)

- exposing specific ports / volumes between the host & docker env.

  - `-p8080:8080 -p8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 `  TorchServe对基于REST的推理、管理和度量API使用默认端口8080 / 8081 / 8082，对gRPC API使用7070/7071。你可能想把这些端口暴露给主机，用于Docker和主机之间的HTTP和gRPC请求。
  - 模型库是通过--模型库选项传递给torchserve的。如果你喜欢在model-store目录中预先填充模型，你可能想考虑使用一个共享卷。

For example,

```shell
docker run --rm --shm-size=1g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p8080:8080 \
        -p8081:8081 \
        -p8082:8082 \
        -p7070:7070 \
        -p7071:7071 \
        --mount type=bind,source=/path/to/model/store,target=/tmp/models <container> torchserve --model-store=/tmp/models
```



### 内网部署的命令测试：

<span style='color:brown'>**示例指令:** </span>

```shell
$ docker run --rm -it --gpus '"device=2"' -p 8082:8080 -p 8081:8081 -v /mnt/cephfs/workspace/codegen-350M/Huggingface_Transformers/model-store:/home/model-server/model-store -v /mnt/cephfs/workspace/codegen-350M/Huggingface_Transformers:/home/Huggingface_Transformers  hub.2980.com/dockerhub/pytorch/torchserve:0.5.3-gpu
```

<span style='color:brown'>**实际使用指令：**</span>

- GPU 部署

  ```shell
  $ docker run --rm -it --gpus '"device=1"' -p 7000:7000 -p 7001:7001 -p 8081:8080 -v /mnt/cephfs/workspace/codegen-350M:/home/codegen-350M hub.2980.com/dockerhub/pytorch/torchserve:latest-gpu
  ```

- CPU部署

  ```shell
  $ docker run --rm -it -p 7000:7000 -p 7001:7001 -p 8081:8080 -v /mnt/cephfs/workspace/codegen-350M:/home/codegen-350M hub.2980.com/dockerhub/pytorch/torchserve:latest-gpu
  ```

- 进入容器内(内网源安装transformers库)

  ```shell
  $ pip install transformers -i https://apt.2980.com/pypi/simple
  # 卸载后安装新版本的transformers库
  $ pip uninstall transformers
  # 进入transformer-main 文件夹进行dev版本库的安装
  $ python setup.py install
  ```

- 重新构建 .mar模型文件

  ```shell
  torch-model-archiver --model-name codegen --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json"
  ```

- 运行模型

  ```shell
  torchserve --start --model-store model-store --models codegen.mar  --ts-config ./config.properties
  ```

- 测试推理的可用性

  ```shell
  $ curl -X POST http://10.17.68.105:7000/predictions/codegen -H 'Content-Type: application/json' -d '{"data":"import matplot"}'
  ```



### <span style='color:brown'>目前的内网机器的服务部署现状：</span>

- 1080GPU(10.17.67.221)机上已验证可以正常推理；
- 3090GPU(10.17.68.105)目前无法进行正常的GPU推理，需要进一步研究

解决方案见03篇文章。





## <span style='color:brown'>问题集</span>



### 1、CUDA error -->  no kernel image available for execution

- [PyTorch Link](https://discuss.pytorch.org/t/need-help-solution-for-cuda-error-no-kernel-image-available-for-execution/146634)
- [Github-issue Link](https://github.com/pytorch/serve/issues/1754)

> 
>
> **Anwser from msaroufim**
>
> Hi [@amanjain1397appy](https://github.com/amanjain1397appy) does your model also fail when you're not serving it with torchserve? I would need more repro instructions which you can see in the bug template we have here on github https://github.com/pytorch/serve/issues/new?assignees=&labels=&template=bug.yml
>
> Regardless my suspicion is this is happening because we don't support pytorch 1.12 in torchserve yet but this was something [@mreso](https://github.com/mreso) had been asking for and [@lxning](https://github.com/lxning) was working on so we'll keep you posted!
>
> If you run this script it should confirm which version of pytorch you actually have installed https://github.com/pytorch/serve/blob/master/ts_scripts/print_env_info.py
>
> If you can't wait feel free to update the version for torch here in the requirements.txt and hopefully the issue just goes away https://github.com/pytorch/serve/tree/master/requirements
>
> 



### 2、Huggingface | ValueError: Connection error, and we cannot find the requested files in the cached path. Please try again or make sure your Internet con

- [Github-Transformers-issues --Link](https://github.com/huggingface/transformers/issues/10067)
- [StackOverflow-Link](https://stackoverflow.com/questions/71335585/huggingface-valueerror-connection-error-and-we-cannot-find-the-requested-fil)



> 
>
> **Answer from LysandreJik**
>
> Could you try to load the model/tokenizer and specify the `local_files_only=True` kwarg to the `from_pretrained` method, before passing them to the pipeline directly?
>
> i.e., instead of:
>
> ```python
> pipeline('sentiment-analysis')('I love you')
> ```
>
> try:
>
> ```python
> from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
> 
> model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", local_files_only=True)
> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", local_files_only=True)
> 
> pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)('I love you')
> 
> ```
>
>

<span style='color:brown'>**此处对应需要更改的部分：**</span>

- Transformer_handler_generalized.py

  ```python
  # 加载新的Tokenizer
  # self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
  self.tokenizer = AutoTokenizer.from_pretrained("/home/codegen-350M/Huggingface_Transformers/Transformer_model/", local_files_only=True)
  ```



**其他可能存在的注意事项：**

- greedy_search 的大小设置
- Inference batch 的相关推理