#!/bin/bash

docker build -t predict-api:gpu -f ../docker/gpu.Dockerfile  --target base ../