# TorchServe 的内网部署实践



**Colab：**

- Colab 不支持docker image build 的相关操作，[Link-1](https://github.com/googlecolab/colabtools/issues/299)；



**构建的镜像应当具备的环境条件：**

1. Python >= 3.8

2. Torchserve

   - cuda

     ```shell
     python ./ts_scripts/install_dependencies.py --cuda=cu102
     ```

   - cpu

     ```shell
     python ./ts_scripts/install_dependencies.py 
     ```

   - torch-model-archiver

     ```shell
     pip install torchserve torch-model-archiver torch-workflow-archiver
     ```

3. Transformers库

   ```python
   pip install transformers
   ```

4. JupyterLab库

   为了方便内网的部署及代码的交互、改写、Dubug的相关便捷性；

   

### <span style='color:brown'>基础镜像的选择</span>

**jupyter/minimal-notebook**

jupyter/minimal-notebook增加了在Jupyter应用程序中工作时有用的命令行工具

- Everything in `jupyter/base-notebook`;
- 用于notebook文档转换的TeX Live;
- [git](https://git-scm.com/), [vi](https://www.vim.org/) (actually `vim-tiny`), [nano](https://www.nano-editor.org/) (actually `nano-tiny`), `tzdata`, and `unzip`

```shell
docker pull jupyter/minimal-notebook:latest
```



### <span style='color:brown'>Dockerfile:</span>

```dockerfile
FROM jupyter/minimal-notebook:latest
# 基础镜像的python == 3.10

USER root
#set TimeZone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai  /etc/localtime && \
	echo "Asia/Shanghai" > /etc/timezone && \
	dpkg-reconfigure -f noninteractive tzdata

# add user
ARG DY_USER="jovyan"
ARG DY_UID="1000"
ARG DY_GID="100"
ARG HOME="/home/jovyan"
ENV DY_USER=$DY_USER \
	DY_UID=$DY_UID \
	DY_GID=$DY_GID	

# RUN pip install -U pip
RUN python -m pip install --upgrade pip

# 1、将本地文件复制到镜像并进行安装操作
COPY ./ts_scripts  /home/jovyan/
# RUN python ./ts_scripts/install_dependencies.py --cuda=cu102
CMD ["/home/jovyan/ts_scripts/install_dependencies.py", "--cuda=cu102"]

# 2、安装 torch-model-archiver
RUN pip install torchserve torch-model-archiver torch-workflow-archiver

# 3、安装 transformers
RUN pip install transformers

# 安装svn功能
RUN apt-get -y update && apt-get install -y subversion

# 安装tini，以免出现僵尸进程
ADD tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# 换成dy1用户
USER $DY_UID
WORKDIR $HOME
```



**启动镜像创建容器的命令**

```shell
docker run -d -it -p 8888:8888 -v /home/workspace:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root jupyter/minimal-notebook:python-3.8.8
```

docker pull jupyter/minimal-notebook:python-3.8.8

```shell
cd your-working-directory 

docker run -d -it -p 8848:8888 -v /home/workspace:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root jupyter/minimal-notebook:latest
```

```shell
cd your-working-directory 

docker run --gpus all -d -it -p 8848:8888 -v $(pwd)/data:/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root cschranz/gpu-jupyter:v1.4_cuda-11.0_ubuntu-20.04_python-only
```

访问地址：http://localhost:8848，进入容器并查看登录 token ：

```shell
$ docker exec -it <container-id> bash
```

```shell
$ jupyter server list
```

获取token后即可登录成功；

在保持容器运行的情况下安全退出容器：

> To **detatch** from the container *without* stopping it press **CTRL+P** followed by **CTRL+Q**.



**构建镜像**

```shell
docker build -t <container-name> .
```

测试Dockerfile的可行性：

```
docker build -t torchserve:test7 .
```



### <span style='color:brown'>**测试构建的镜像的可用性**</span>

镜像名称：

- torchserve-deploy:test2

启动镜像：

```shell
docker run -it -p 8889:8888 -v /home/workspace:/home/jovyan/work  -e JUPYTER_ENABLE_LAB=yes torchserve:test6
```





<span style='color:brown'>**最新的Dockerfile文件：**</span>

```shell
FROM jupyter/minimal-notebook:python-3.8.8
# 基础镜像的python == 3.8

USER root
#set TimeZone
# RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai  /etc/localtime && \
# 	echo "Asia/Shanghai" > /etc/timezone && \
# 	dpkg-reconfigure -f noninteractive tzdata

# # add user
# ARG DY_USER="jovyan"
# ARG DY_UID="1000"
# ARG DY_GID="100"
ARG HOME="/home/jovyan"

# ENV DY_USER=$DY_USER \
# 	DY_UID=$DY_UID \
# 	DY_GID=$DY_GID	


# # 修正权限问题，让jovyan可以用pip安装python包
# ADD fix-permissions /usr/local/bin/fix-permissions
# RUN chmod 755 /usr/local/bin/fix-permissions

# RUN fix-permissions /usr/local/bin && \
#     fix-permissions /usr/local/lib && \
#     fix-permissions /usr/local/include && \
#     fix-permissions /usr/local/share

WORKDIR $HOME

# RUN pip install -U pip
RUN pip install --upgrade pip

# 1、将本地文件复制到镜像并进行安装操作
COPY ./ts_scripts/  /home/jovyan/ts_scripts/
COPY ./requirements/  /home/jovyan/requirements/ 
RUN python ./ts_scripts/install_dependencies.py  --cuda=cu102

# RUN python ./ts_scripts/install_dependencies.py --cuda=cu102
# CMD ["/home/jovyan/ts_scripts/install_dependencies.py", "--cuda=cu102"]

# 2、安装 torch-model-archiver
RUN pip install  --no-cache-dir  torchserve torch-model-archiver torch-workflow-archiver

# # 3、安装 transformers
# RUN pip install  --no-cache-dir  transformers

# # 安装svn功能
# RUN apt-get -y update && apt-get install -y subversion

# 安装tini，以免出现僵尸进程
# ENV TINI_VERSION v0.19.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini

# ADD tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini

# 换成dy1用户
# USER $DY_UID
# WORKDIR $HOME
```

