ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 sox vim\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.2.0+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html


RUN conda clean --all
WORKDIR /mmcaptioning

ENV FORCE_CUDA="1"

RUN pip install cython==0.29.21
RUN pip install pandas==1.1.4
RUN pip install flask==1.1.2 flask_cors==3.0.9
RUN pip install flask_restful_swagger_2==0.35
RUN pip install librosa==0.8.0
RUN pip install mmpycocotools==12.0.3
RUN pip install pycocotools==2.0.2
RUN pip install transformers==3.4.0
RUN pip install kobert-transformers==0.4.1
RUN pip install google-cloud-speech==2.0.0
