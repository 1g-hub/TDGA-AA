# AutoAugment with TDGA

Experiment for AutoAugment with TDGA.

## Requirements
- [Docker](https://www.docker.com/) >= 19.03
- [GNU Make](https://www.gnu.org/software/make/)
- [nvidia-drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) (Only for GPU)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (Only for GPU)

## Installation
### Clone repository
```bash
$ git clone https://github.com/1g-hub/TDGAAutoAugment
$ cd TDGAAutoAugment
```

### Build image
```bash
$ make build
```

**NOTE:** <br>
If you want use GPUs, install [nvidia-drivers](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) first.

## How to Use

### run the container
```
$ make bash
```

