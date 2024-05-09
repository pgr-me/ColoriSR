FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install vim git wget -y

# Install libraries
RUN conda install -y -c conda-forge jupyter pandas matplotlib tqdm scikit-image
RUN conda install -y -c fastai fastai
RUN pip install torchmetrics[image]


# Copy over preprocessing script 
COPY preprocess_data.py /workspace/
COPY color_to_sr/ /workspace/

