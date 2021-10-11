# use the name of the current directory as the docker image tag
DOCKERFILE ?= Dockerfile
DOCKER_USERNAME = marcozecchini
DOCKER_REPO ?= $(shell echo ${PWD} | rev | cut -d/ -f1 | rev)
DOCKER_TAG = latest
DOCKER_IMAGE = ${DOCKER_USERNAME}/${DOCKER_REPO}:${DOCKER_TAG}


.PHONY: image
image: requirements.txt
	docker build \
        -t ${DOCKER_IMAGE} \
        -f ${DOCKERFILE} \
        .

.PHONY: run
run:
	docker run \
         -t $(DOCKER_IMAGE)

.PHONY: run_background
run_background:
	docker run -d  \
         -t $(DOCKER_IMAGE) \


test:
	python train.py