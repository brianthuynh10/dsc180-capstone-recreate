# Base image: includes Python, CUDA, cuDNN, and PyTorch preinstalled
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Install basic system tools and update pip
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential && \
    pip install --upgrade pip

# Install your main Python libraries
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    torch \
    torchvision \
    wandb \
    scipy \
    h5py \
    tqdm 
    

# Set a working directory inside the container
WORKDIR /workspace

# Default command — drop you into a bash shell
CMD ["bash"]