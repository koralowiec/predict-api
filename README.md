# Serving Object Detection model from TensorFlow Hub with FastAPI

Tested on: 
 - Docker version: 19.03.12-ce
 - Docker Compose version: 1.26.1

## Clone

```bash
git clone https://github.com/koralowiec/predict-api
cd predict-api
```

## Downloading the TensorFlow Hub's model

The model can be downloaded (to /tmp) and untared (to ./modules) with simple script:

```bash
cd ./scripts
./download-module-and-untar-to-tmp.sh
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


## Local development with docker-compose

Nvidia Container Toolkit doesn't work with docker-compose yet ([Github issue](https://github.com/docker/compose/issues/6691)), it's needed to use nvidia-docker: [Docker Compose support](https://github.com/NVIDIA/nvidia-docker/wiki#do-you-support-docker-compose)

After installing nvidia-docker, restart a docker service and check if nvidia is shown as runtime:

```bash
sudo systemctl restart docker.service
docker info | grep Runtimes
```

Then to run container with automatic restart after every save in index.py type:

```bash
docker-compose -f ./docker/docker-compose.dev.yml up
```

## Running

### Run with Docker

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
2. On GPU (needed installed Nvidia driver, [more information](https://www.tensorflow.org/install/docker#gpu_support))
	1. Build an image:
		```bash
		cd ./scripts
		./build-docker-image-gpu.sh
		```
	2. Run a container:
        ```bash
		./run-docker-gpu.sh
		```
	3. Possible troubles:
		- `E tensorflow/stream_executor/cuda/cuda_dnn.cc:329] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR`
		If something like this appears in logs, stop container and start new one.

### Run without Docker

1. Create and activate virtual environment with virtualenv:
	```bash
	virtualenv -p python3 env
    source ./env/bin/activate
	```
2. Install dependencies:
   	```bash
    pip install tensorflow==2.2.0
    pip install -r requirements.txt
	```
3. Manually change line 36 in main.py file with correct path to model (if you used script from the first step it should be: ./modules/openimages_v4__ssd__mobilenet_v2):
   	```python
	tf_hub_module = hub.load(module_path).signatures["default"]
	```
4. Run a server:
	```bash
	uvicorn --app-dir code main:app --port 5000 --host 0.0.0.0
	```