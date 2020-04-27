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


#RUN apt-get update && apt-add-repository -y ppa:jonathonf/ffmpeg-3 && apt-get update

#RUN apt-get install -y ffmpeg libav-tools x264 x265

RUN apt-get update && apt-get install -y libsm6 libxrender1 libxext-dev

RUN apt-get -y clean all

RUN rm -rf /var/lib/apt/lists/*

# Install some useful and machine/deep-learning-related packages for Python3.

RUN pip install keras==2.1.5
RUN pip install scikit-image
RUN pip install opencv-python

# RUN conda install tensorboard 
# RUN conda install -c conda-forge av tensorboardx

#RUN conda install jupyterlab
#RUN conda install jupyterlab_templates

## OPENCV ##
#RUN conda install opencv-contrib-python
#RUN conda install -c conda-forge opencv
#RUN conda install av
#RUN conda install -c conda-forge tensorboardx

# RUN curl -L  "https://oscdl.ipa.go.jp/IPAexfont/ipaexg00301.zip" > font.zip
# RUN unzip font.zip
# RUN cp ipaexg00301/ipaexg.ttf /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/fonts/ttf/ipaexg.ttf
# RUN echo "font.family : IPAexGothic" >>  /usr/local/lib/python3.5/dist-packages/matplotlib/mpl-data/matplotlibrc
# RUN rm -r font.zip ipaexg00301

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

#RUN jupyter labextension install jupyterlab_templates
#RUN jupyter labextension install @lckr/jupyterlab_variableinspector
USER yoshida
#RUN jupyter serverextension enable --py jupyterlab
#RUN jupyter notebook --generate-config
