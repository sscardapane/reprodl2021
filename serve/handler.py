import io
import os
import pandas as pd


import torch, torchaudio
from torch import nn
from torch.nn import functional as F
from miniaudio import SampleFormat, decode
import soundfile as sf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


import hydra
from hydra.experimental import compose
from hydra.experimental import initialize as init_cfg
from omegaconf import DictConfig, OmegaConf


import json


import logging
logger = logging.getLogger(__name__)


from ts.torch_handler.base_handler import BaseHandler

from model import AudioNet






class MyHandler(object):
    
    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False


    def initialize(self, ctx): 

        properties = ctx.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        model_dir = properties.get("model_dir")
        model_pt_path = os.path.join(model_dir, "model.pth")

        with open('index_to_name.json') as json_file:
            self.mapping  = json.load(json_file)


        init_cfg(config_path="./", job_name="test_app")
        self.cfg = compose(config_name="default")

        self.model = AudioNet(self.cfg.model)

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True



    def preprocess(self, data):

        audio = data[0].get("data")

        if audio is None:
            audio = data[0].get("body")
        
        resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=8000)
        melspec = torchaudio.transforms.MelSpectrogram(sample_rate=8000)
        db = torchaudio.transforms.AmplitudeToDB(top_db=80)


        wav, samplerate = sf.read(io.BytesIO(audio))

        wav = torch.FloatTensor(wav)

        xb = resample(wav.unsqueeze(0))
        xb = melspec(xb)
        xb = db(xb)

        return  xb.unsqueeze(0)



    def inference(self, audio):

        self.model.eval()
        y_pred = self.model.forward(audio)
        predicted_idx = y_pred.argmax(-1).item()

        return [str(predicted_idx)]



    def postprocess(self, inference_output):
        res = []
        for pred in inference_output:
            label = self.mapping[str(pred)][1]
            label = '\n\t'+label+'\n\n'
            res.append(label)
        return res








_service = MyHandler()




def handle(data, context):

    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)

    data = _service.inference(data)

    data = _service.postprocess(data)

    return data



