#! /bin/bash
# run container
docker run -p 6007:22 \
           --gpus all \
           --ipc=host \
           --rm -dit \
	   --name nvidia \
	   --workdir /root/code \
           -v /home/freddy/code:/root/code \
           nvcr.io/nvidia/pytorch:"$1"-py3

# copy IDE configurations
docker cp -q /home/freddy/.cache/JetBrains nvidia:/root/.cache/JetBrains

# return to docker terminal
docker attach nvidia
