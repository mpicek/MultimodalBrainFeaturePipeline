# Use the base image provided by DeepLabCut
FROM deeplabcut/deeplabcut:2.3.5-base-cuda11.7.1-cudnn8-runtime-ubuntu20.04-latest

# Install required Python packages
RUN pip install jupyter ffmpeg-python

# Expose the Jupyter port
EXPOSE 8888

# Set the working directory
WORKDIR /workspace

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

