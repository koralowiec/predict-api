# Serving Object Detection model from TensorFlow Hub with FastAPI

Tested on: 
 - Docker version: 19.03.12-ce
 - Docker Compose version: ≥1.26.1

## Clone

```bash
git clone https://github.com/koralowiec/predict-api
cd predict-api
```

## Downloading the TensorFlow Hub's model

The models can be downloaded (to /tmp) and untared (to ./modules) with simple script:

```bash
cd ./scripts
./download-module-and-untar.sh
```

You can choose to download only one of two models:

```bash
# openimages_v4__ssd__mobilenet_v2
./download-module-and-untar.sh -m 1
# faster_rcnn_openimages_v4_inception_resnet_v2
./download-module-and-untar.sh -m 2
```

If something like that appears:

```bash
bash: ./download-module-and-untar-to-tmp.sh: Permission denied
```

You need add execution right:

```bash
chmod u+x ./download-module-and-untar-to-tmp.sh
```

And try running once again

## Run with Docker

For running on (Nvidia) GPU with Docker, its needed to install Nvidia driver and [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker) on host. More information may be found on TensorFlow's documentation [here](https://www.tensorflow.org/install/docker#gpu_support)

### Image from GitHub Container Registry

Built images can be pulled using commands:

```bash
# image for running on CPU
docker pull ghcr.io/koralowiec/predict-api:cpu
# image for running on GPU
docker pull ghcr.io/koralowiec/predict-api:gpu
```

### Build image on your own

1. On CPU
	1. Build an image:
		```bash
		cd ./scripts
		./build-docker-image-cpu.sh
		```
	2. Run a container:
        ```bash
		./run-docker-cpu.sh
		```
2. On GPU
	1. Build an image:
		```bash
		cd ./scripts
		./build-docker-image-gpu.sh
		```
	2. Run a container:
        ```bash
		./run-docker-gpu.sh
		```

### Run container

```bash
# from root of project

# CPU image with openimages_v4__ssd__mobilenet_v2
docker run -p 5002:5000 -v $(pwd)/modules/openimages_v4__ssd__mobilenet_v2:/model_ssd ghcr.io/koralowiec/predict-api:cpu

# CPU image with faster_rcnn_openimages_v4_inception_resnet_v2
docker run -p 5002:5000 -v $(pwd)/modules/faster_rcnn_openimages_v4_inception_resnet_v2:/model_faster_rcnn -e MODULE=FASTER_RCNN ghcr.io/koralowiec/predict-api:cpu

# GPU image with faster_rcnn_openimages_v4_inception_resnet_v2
docker run --gpus all -p 5002:5000 -v $(pwd)/modules/faster_rcnn_openimages_v4_inception_resnet_v2:/model_faster_rcnn -e MODULE=FASTER_RCNN ghcr.io/koralowiec/predict-api:gpu
```

Possible error while running on GPU:

`E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR`

Solution: restart container


## Local development with docker-compose

### CPU

To run container with automatic restart after every save in *.py files type:

```bash
docker-compose -f ./docker/docker-compose.dev.yml up
```

### GPU

Nvidia Container Toolkit doesn't work with docker-compose yet ([Github issue](https://github.com/docker/compose/issues/6691)), it's needed to use older nvidia-docker2: [Docker Compose support](https://github.com/NVIDIA/nvidia-docker/wiki#do-you-support-docker-compose)

After installing nvidia-docker, restart a docker service and check if nvidia is shown as runtime:

```bash
sudo systemctl restart docker.service
docker info | grep Runtimes
```

Then to run container with automatic restart after every save in *.py files type:

```bash
docker-compose -f ./docker/docker-compose.gpu.dev.yml up
```

## Run without Docker

1. Create and activate virtual environment:
	```bash
	python -m venv env
    source ./env/bin/activate
	```
2. Install dependencies:
   	```bash
    pip install tensorflow==2.2.1
    pip install -r requirements.txt
	```
3. Manually change line 36 in main.py file with correct path to model (if you used script for downloading model it should be: ./modules/openimages_v4__ssd__mobilenet_v2):
   	```python
	tf_hub_module = hub.load(module_path).signatures["default"]
	```
4. Run a server:
	```bash
	uvicorn --app-dir code main:app --port 5000 --host 0.0.0.0
	```