# Reproducible Deep Learning
## Extra: TorchServe
### Author: [FraLuca](https://github.com/FraLuca)

Before starting, read the main documentation for [Torchserve](https://pytorch.org/serve/) from the Pytorch website.

> :warning: **extra** branches implement additional exercises created by the students of the course to explore additional libraries and functionalities. They can be read independently from the main branches. Refer to the original authors for more information.

&nbsp;


## Prerequisites

1. Uncompress the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) inside the *data* folder.
2. Install requirements (Java and Python libraries):

```bash
sudo apt install --no-install-recommends -y openjdk-11-jre-headless
pip install torchserve==0.2.0 torch-model-archiver==0.2.0
pip install SoundFile==0.10.3.post1
```

&nbsp;


## Goal

The aim of this exercise is to understand how to serve a PyTorch model. In this way, every user can easily use a pre-trained code without the need for training once again from scratch, and without having to adapt the code for a simple evaluation on a test data sample.

With these istructions we will see how to deploy model serving and how much is it easy to use for inference.


&nbsp;

## Instructions


1. Add to the `train.py` script a line to save your model once it is trained.

```python
torch.save(trainer.model.state_dict(), "model.pth")
```

In this way we can load the learned parameters and use them to evaluate the model on new data.


**Note:** We need to move all the scripts/folders/data (e.g., `model.pth`) to the folder "serve".

&nbsp;

2. Then we create an instance of the model by copy-pasting the class `AudioNet` from `train.py` to a new file named `model.py`.

> :speech_balloon: This step is not mandatory, but allows you to make things more orderly for possible future changes.


&nbsp;

3. Create a handler class in a `handler.py` script. This is the core of this tool.
  Basically here we define a class with 4 main functions:
    - `initialize`: here we instantiate and load the pre-trained model. Moreover we can set other case-specific settings;
    - `preprocess`: this function receives the data and then preprocess them before feeding them to the net;
    - `inference`: where the model extracts the prediction;
    - `postprocess`: some final postprocessing operations, like mapping the index of the class to the corresponding class name.


&nbsp;


4. In the same file (`handler.py`) define a function named `handle` that instanciates the previous class and uses it.

```python
_service = MyHandler()

_service.initialize(context)

data = _service.preprocess(data)

data = _service.inference(data)

data = _service.postprocess(data)
```

Its inputs must be:
  - `data`: data element, generally as `bytearray`;
  - `context`: item that contains some environment info (e.g. presence of gpus or work directory).


&nbsp;


5. Add the `index_to_name.json` file that will map the predicted index class for your test sample to its string name.

> :speech_balloon: We provide a `gen_dict.py` file to create the json for this specific case.

&nbsp;

6. Create a test_data folder where to move the data you want to infer.


> :speech_balloon: To be clear all the files and folders created until now MUST be in the serve folder. This is a list of what it has to contain:
>   - `test_data` (folder)
>   - `handler.py`
>   - `model.py`
>   - `index_to_name.json`
>   - `model.pth`
>   - `gen_dict` (Optional)

&nbsp;

7. Now we are ready to serve our model with TorchServe! Here we will generate the .mar file which contains the archived version of the model. So we create a folder "model_store" to store it and then we create the AUDIONET archived model. 

```bash
mkdir serve/model_store

torch-model-archiver --model-name AUDIONET --version 1.0 --serialized-file ./serve/model.pth --model-file ./serve/model.py --handler ./serve/handler.py --export-path ./serve/model_store -f --extra-files ./configs/default.yaml,./serve/index_to_name.json
```

&nbsp;

8. At the end we deploy the Torchserve REST API on localhost ready to serve.

```bash
torchserve --start --model-store ./serve/model_store --models audionet=AUDIONET.mar --no-config-snapshots
```

&nbsp;

9. On a new terminal (that points at the same folder of the previous one and with the right conda environment activated) we can test our data sample.

```bash
curl -X POST http://127.0.0.1:8080/predictions/audionet -T ./serve/test_data/1-14262-A-37.wav
```

&nbsp;

10. Then we can try with all the samples we want. Finally we can stop the API, on the same terminal where we started it, with the following command:

```bash
torchserve --stop
```

&nbsp;

> :speech_balloon: All the files created in the steps 1-7 are included in the repo. So you can start directly testing the trained classifier following points 8-10! This is the advantage of serving your model.


&nbsp;

Congratulations! You have concluded this extra exercise. :nerd_face:


