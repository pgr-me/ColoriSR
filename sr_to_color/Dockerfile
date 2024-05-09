FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update && apt-get install vim git wget -y

# Install libraries
RUN conda install -y -c conda-forge jupyter pandas matplotlib tqdm scikit-image opencv
RUN conda install -y -c fastai fastai
RUN pip install basicsr facexlib gfpgan
RUN git clone https://github.com/xinntao/Real-ESRGAN.git
WORKDIR /workspace/Real-ESRGAN/
RUN python setup.py develop

# Download pretrained models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth  -P experiments/pretrained_models

# Copy over preprocessing script and config file
COPY preprocess_data.py /workspace/Real-ESRGAN/
COPY sr_to_color/finetune_colorizer_sr.yml /workspace/Real-ESRGAN/options/

