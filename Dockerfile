FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential vim git wget curl unzip software-properties-common libgl1-mesa-glx
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.9 python3.9-distutils python3.9-dev
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN wget -N https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN pip install --upgrade pip
RUN rm get-pip.py


RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
RUN pip install opencv-python
RUN pip install numpy==1.24.1
RUN pip install mss tqdm

RUN apt-get update && apt-get install -y xvfb x11-utils x11-xserver-utils x11-apps xrandr