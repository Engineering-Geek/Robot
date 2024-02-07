# Driver Version: 535.154.05   CUDA Version: 12.2
# Cuda compilation tools, release 12.0, V12.0.140
FROM nvcr.io/nvidia/pytorch:24.01-py3
LABEL authors="engineering-geek"

# Define an environment variable for the destination path
ENV DESTINATION_PATH /your/destination/path

# Update and upgrade the system
RUN apt update && apt upgrade -y

# Install required packages
RUN apt install -y libglfw3-dev libglfw3 git ffmpeg

# Install Python packages using pip
RUN pip install --upgrade pip
RUN pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install mujoco mujoco_mjx brax pyyaml matplotlib mediapy

# Clone the Mujoco Menagerie repository to the destination path
RUN git clone https://github.com/google-deepmind/mujoco_menagerie.git $DESTINATION_PATH

# Set the working directory to the cloned repository
WORKDIR $DESTINATION_PATH
