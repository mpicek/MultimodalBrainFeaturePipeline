# Docker for synchronization

Includes **DeepLabCut** docker with additional python packages for synchronizing videos with the brain signals.
Nvidia might have problems when the docker is installed via Snap (it's better to uninstall it
with `sudo snap remove docker --purge` and [install Docker with apt](https://docs.docker.com/engine/install/ubuntu/) if you are on Ubuntu).

**Prerequisities**: [Install nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

To build the image, run:
```
sudo docker build -t deeplabcut-with-jupyter .
```

To run, run (on my machine):
```
xhost + # to allow forwarding of the visual output from the docker

# then enable the Persistence Mode on all the GPUs available on my machine
sudo nvidia-smi -i 0 -pm ENABLED
sudo nvidia-smi -i 1 -pm ENABLED
sudo nvidia-smi -i 2 -pm ENABLED

# and finally run the docker forwarding the output (-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network="host"),
# enabling the gpus (--gpus all), forwarding the jupyter notebook port (-p 8888:8888 ) and mounting volumes (the first three -v arguments)
sudo docker run -v /home/vita-w11/mpicek/:/mpicek/ -v /home/vita-w11/Downloads/:/data/ -v /home/vita-w11/mpicek/master_project/:/repo/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 --network="host" -it deeplabcut-with-jupyter
```