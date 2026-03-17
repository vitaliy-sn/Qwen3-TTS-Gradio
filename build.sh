#!/bin/bash

tag="latest"
cuda_version="$(cat Dockerfile | grep "^ARG CUDA_VERSION" | awk -F'[=.]' '{print $2"."$3}')"

docker_tag="qwen3tts:${tag}-cuda-${cuda_version}-ubuntu24.04"

if [ "$tag" == "" ]; then
  exit 1
fi

if [ "$cuda_version" == "" ]; then
  exit 1
fi

docker buildx build --progress=plain --build-arg PIP_INDEX_URL=http://10.4.1.142:3141/root/pypi/ --build-arg PIP_TRUSTED_HOST=10.4.1.142 . --tag ${docker_tag}

echo "$docker_tag"
