FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel AS env

# Set the working directory
WORKDIR /reprodl 

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install libsndfile1-dev -y

FROM env

# Copy the files
COPY . ./

# Install DVC
RUN pip install dvc boto3 --ignore-installed ruamel.yaml

# Set the access keys
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Download the data
RUN dvc pull

CMD ["python", "train.py", "~trainer.gpus"]

