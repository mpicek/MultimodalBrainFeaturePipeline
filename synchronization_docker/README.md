# Docker for synchronizationIncludes **DeepLabCut** docker with additional python packages for synchronizing videos with the brain signals.

Nvidia might have problems when the docker is installed via Snap (it's better to uninstall it
with `sudo snap remove docker --purge` and [install Docker with apt](https://docs.docker.com/engine/install/ubuntu/) if you are on Ubuntu).

**Prerequisities**: [Install nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

To build the image, run:
```
sudo docker build -t deeplabcut-with-jupyter .
```
To run, run (on my machine):
```
xhost + # to allow forwarding of the visual output from the docker# then enable the Persistence Mode on all the GPUs available on my machine
sudo nvidia-smi -i 0 -pm ENABLED
sudo nvidia-smi -i 1 -pm ENABLED
sudo nvidia-smi -i 2 -pm ENABLED
```
And finally run the docker forwarding the output (-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network="host"), enabling the gpus (--gpus all), forwarding the jupyter notebook port (-p 8888:8888 ) and mounting volumes (the first three -v arguments). **This is, however, for creating the jupyter notebook! There, you can train the network:**

```
sudo docker run -v /home/vita-w11/mpicek/:/mpicek/ -v /home/vita-w11/Downloads/:/data/ -v /home/vita-w11/mpicek/MultimodalBrainFeaturePipeline/:/repo/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 --network="host" -it deeplabcut-with-jupyter
```
**But to run only the processing of files (/data_synchronization/DLC/analyze_video.py), you have to run:**

```
sudo docker run -v /home/vita-w11/mpicek/:/mpicek/ -v /home/vita-w11/Downloads/:/data/ -v /home/vita-w11/mpicek/MultimodalBrainFeaturePipeline/:/repo/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 --network="host" -it deeplabcut-with-jupyter bash
```
And probably change some mounting (for example data location will be different etc.). Then go to repo, find `analyze_video.py` and run it :)