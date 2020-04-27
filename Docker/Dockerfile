FROM nvcr.io/nvidia/tensorflow:19.07-py3

MAINTAINER Soh Yoshida <sohy@kansai-u.ac.jp>


ARG NUM_CPUS_FOR_BUILD=20

RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo
RUN apt-get update \
  && apt-get install -y tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && echo "${TZ}" > /etc/timezone \
  && rm /etc/localtime \
  && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
  && dpkg-reconfigure -f noninteractive tzdata


RUN apt-get update && apt-get install -y libsm6 libxrender1 libxext-dev
RUN apt-get -y clean all
RUN rm -rf /var/lib/apt/lists/*


RUN pip install keras==2.1.5
RUN pip install scikit-image
RUN pip install opencv-python

#
ENV USER yoshida
ENV HOME /home/${USER}
ENV SHELL /bin/bash

#
RUN groupadd -g 1001 yoshida
RUN useradd -g 100 -u 1001 -m -s /bin/bash ${USER}
#
RUN gpasswd -a ${USER} sudo
#
RUN echo "${USER}:yoshida" | chpasswd

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo '${USER} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER yoshida
