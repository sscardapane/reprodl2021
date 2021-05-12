# Reproducible Deep Learning
## Exercise 4: Dockerization
[[Official website](https://www.sscardapane.it/teaching/reproducibledl/)] [[Slides](https://docs.google.com/presentation/d/1r7SbbajL-UnYHOeY9fQ9YtoJdu9Q70U5M_11E68K1Rg/edit?usp=sharing)] [[Docker Website](http://dvc.org/)]

## Objectives for the exercise

- [ ] Pulling images and running containers.
- [ ] Building custom images from Dockerfiles.
- [ ] Pushing/pulling images from the Docker Hub.

See the completed exercise:

```bash
git checkout exercise4_docker_completed
```

## Prerequisites

1. Complete [Exercise 3](https://github.com/sscardapane/reprodl2021/tree/exercise3_dvc). Leave the MinIO server in execution.
2. Install [Docker Desktop](https://docs.docker.com/get-docker/). For using a CUDA-enabled GPU, install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).
3. If you are using Visual Studio Code, install the [Docker extension](https://code.visualstudio.com/docs/containers/overview).

Before starting, check that Docker is installed correctly:

```bash
docker run hello-world
```

## Preliminaries

[Requirement specifiers](https://pip.pypa.io/en/stable/cli/pip_install/#requirement-specifiers) are very useful text files detailing all the Python libraries needed for a project. Let us create one!

1. Install `pipreqs`:

```bash
pip install pipreqs
```

2. Generate a requirements.txt file from the current directory:

```bash
pipreqs .
```
3. While powerful, `pipreqs` is not perfect, which is why you should look carefully at the generated file. In particular, at the moment [it sets the wrong dependencies for Hydra](https://github.com/bndr/pipreqs/issues/244). __Remove `hydra` from requirements.txt__ (the correct package is `hydra-core`).

### Optional: install a Docker dashboard

It is a good idea to have a decent dashboard to more easily manipulate images and containers. On some systems, you can use the [Docker Dashboard](https://docs.docker.com/desktop/dashboard/). Alternatively, you can use the VS Code plugin.

For a more powerful alternative, you can start a [Portainer CE](https://documentation.portainer.io/quickstart/) container, which is a good exercise in understanding Docker:

```bash
docker volume create portainer_data
docker run -d -p 9001:9000 --name=portainer -v /var/run/docker.sock:/var/run/docker.sock -v portainer_data:/data portainer/portainer-ce
```

> :speech_balloon: We are using port 9001 to avoid conflicts with MinIO on port 9000.

From [localhost:9001](http://localhost:9001) you can now monitor your Docker installation (see the [initial setup](https://documentation.portainer.io/v2.0/deploy/initial/)).

## Dockerfile \#1: developing in a container

Our first Dockerfile will be a working environment inside which to develop (and launch) the application. See the [slides](https://docs.google.com/presentation/d/1r7SbbajL-UnYHOeY9fQ9YtoJdu9Q70U5M_11E68K1Rg/edit?usp=sharing) for a tutorial on building Dockerfiles.

The first Dockerfile `Dockerfile` should:

1. Inherit from the official images of [PyTorch](https://hub.docker.com/r/pytorch/pytorch) or [PyTorch Lightning](https://hub.docker.com/r/pytorchlightning/pytorch_lightning).
2. Install all the required libraries from the requirements file.
3. Install [libsdnfile](https://packages.debian.org/sid/libsndfile1).

Once the Dockerfile is completed, build the image:

```bash
docker build . --rm -t reprodl/env
```

Run the environment in interactive mode:

```bash
docker run -d -it -p 9002:9000 reprodl/env
```
> :speech_balloon: Adding a volume is also a good idea at this point. You can avoid mapping the port if you do not plan to use MinIO.

Find the container ID running `docker ps`, and try attaching to the container running `docker attach <id>`. Alternatively, you can [attach to a running container](https://code.visualstudio.com/docs/remote/attach-container) from Visual Studio Code.

From inside the container, try to replicate a training run.

## Dockerfile \#2: an executable container

Read about [multi-stage builds](https://docs.docker.com/develop/develop-images/multistage-build/). Add a second stage to the Dockerfile that automatically copies all the required data and launches a training run.

1. You can use the [COPY command](https://docs.docker.com/engine/reference/builder/#copy) to copy .py and .yaml files.
2. Pass the MinIO keys using the [ARG command](https://docs.docker.com/engine/reference/builder/#arg).
3. Launch `dvc pull` during the build.
4. Launch the final training command at the end of the Dockerfile with the [CMD command](https://docs.docker.com/engine/reference/builder/#cmd).

Build the new Dockerfile:

```bash
docker build . --rm --build-arg AWS_ACCESS_KEY_ID="minioadmin" --build-arg AWS_SECRET_ACCESS_KEY="minioadmin" -t reprodl/train
```

Launch another container:

```bash
docker run reprodl/train
```

> :speech_balloon: Change the access keys according to your installation.

## Optional: Push / pull from the Docker Hub

You can also try [pushing and pulling](https://docs.docker.com/docker-hub/) your image using Docker Hub. First, tag your image accordingly:

```bash
docker tag reprodl/env <docker-username>/<project>
```

Then, push the image to the Docker Hub:

```bash
docker push <docker-username>/<project>
```

Congratulations! You have concluded another move to a reproducible deep learning world. :nerd_face:

Move to the next exercise:

```bash
git checkout exercise5_wandb
```

### Optional: Only for the DGX machine

You might experience permission issues when building images and running containers on the DGX machine. You can solve most of them by adding these commands to your Dockerfile:

```docker
ARG USER_ID
ARG GROUP_ID
ENV USERNAME=<your-username>

RUN addgroup --gid $GROUP_ID $USERNAME
RUN adduser --home /home/$USERNAME --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_I $USERNAME
USER $USERNAME
```

When building the containers, add the flags `--build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)`. When running an image, instead, add the flag `--user "$(id -u):$(id -g)"`.
