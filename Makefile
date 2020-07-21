NVIDIA_SMI_PATH := $(shell which nvidia-smi)
IMAGE_NAME := python/tdgaautoaugment
CONTAINER_NAME := python.pytorch.tdgaautoaugment
WORKINGDIR := /var/www
PWD := $(shell pwd)

ifdef NVIDIA_SMI_PATH
    DOCKER_GPU_PARAMS := --gpus all
endif

.PHONY: build
build:
	@docker build --tag $(IMAGE_NAME) -f $(PWD)/docker/Dockerfile .

.PHONY: run
run:
	@docker run \
		--rm -it \
		$(DOCKER_GPU_PARAMS) \
		--name $(CONTAINER_NAME) \
		--volume $(PWD):$(WORKINGDIR) \
		--shm-size 32G \
		$(IMAGE_NAME) \
		$(ARGS)
.PHONY: run/root
run/root:
	@docker run \
		--rm -it \
		-u root \
		$(DOCKER_GPU_PARAMS) \
		--name $(CONTAINER_NAME) \
		--volume $(PWD):$(WORKINGDIR) \
		--shm-size 32G \
		$(IMAGE_NAME) \
		$(ARGS)


.PHONY: bash
bash: ARGS=bash
export ARGS
bash:
	@$(MAKE) run

.PHONY: bash/root
bash/root: ARGS=bash
export ARGS
bash/root:
	@$(MAKE) run/root

.PHONY: test
test: ARGS=pytest tests
export ARGS
test:
	@$(MAKE) run

.PHONY: lint
lint: ARGS=flake8 src tests
export ARGS
lint:
	@$(MAKE) run
