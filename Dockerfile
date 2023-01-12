FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install python \
    apt-utils \
    python3-pip \
    python-dev \
    git vim \
    openssh-server


RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN pip install --upgrade pip
RUN pip install setuptools
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /workspace
ADD . .

ENV PYTHONPATH $PYTHONPATH:/workspace

# RUN pip install -r requirements.txt

RUN chmod -R a+w /workspace
