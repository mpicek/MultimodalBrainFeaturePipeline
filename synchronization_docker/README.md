# Docker for synchronization

Includes **DeepLabCut** docker with additional python packages for synchronizing videos with the brain signals.

To build, run:
```
sudo docker build -t deeplabcut-with-jupyter .
```

To run, run (on my machine):
```
xhost + # to allow forwarding of the visual output from the docker

# then enable all the GPUs available on my machine
sudo nvidia-smi -i 0 -pm ENABLED
sudo nvidia-smi -i 1 -pm ENABLED
sudo nvidia-smi -i 2 -pm ENABLED

# and finally run the docker forwarding the output (-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network="host"),
# enabling the gpus (--gpus all), forwarding the jupyter notebook port (-p 8888:8888 ) and mounting volumes (the first three -v arguments)
sudo docker run -v /home/vita-w11/mpicek/:/mpicek/ -v /home/vita-w11/Downloads/:/data/ -v /home/vita-w11/mpicek/master_project/:/repo/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --gpus all -p 8888:8888 --network="host" -it deeplabcut-with-jupyter
```

