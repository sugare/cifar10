# coding: utf-8

import sys
import os
import numpy as np
import grpc
from api import api_pb2
from api import grpc_service_pb2
from api import grpc_service_pb2_grpc
import api.model_config_pb2 as model_config
import librosa # pip install librosa
from tqdm import tqdm # pip install tqdm

# 0 = air_conditioner
# 1 = car_horn
# 2 = children_playing
# 3 = dog_bark
# 4 = drilling
# 5 = engine_idling
# 6 = gun_shot
# 7 = jackhammer
# 8 = siren
# 9 = street_music

typeDict = {
    0:"air_conditioner",
    1:"car_horn",
    2:"children_playing",
    3:"dog_bark",
    4:"drilling",
    5:"engine_idling",
    6:"gun_shot",
    7:"jackhammer",
    8:"siren",
    9:"street_music"
}


url='39.99.222.104:30712'  # http://39.99.222.104:31380/v2
if len(sys.argv)>1:
    print (sys.argv[1])

#####################
model_name='default'
#####################

def parse_model(status, model_name):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    server_status = status.server_status

    status = server_status.model_status[model_name]
    config = status.config

    if len(config.input) != 1:
        raise Exception("expecting 1 input, got {}".format(len(config.input)))

    input = config.input[0]
    output = [i.name for i in config.output]

    return input.name, output

def extract_features(wav_files):
    inputs = []
    labels = []

    for wav_file in tqdm(wav_files):
        # 读入音频文件
        audio, fs = librosa.load(wav_file)

        # 获取音频mfcc特征
        # [n_steps, n_inputs]
        mfccs = np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=40), [1, 0])
        inputs.append(mfccs.tolist())
        # 获取label
    for wav_file in wav_files:
        label = wav_file.split('/')[-1].split('-')[1]
        labels.append(label)
    return inputs, np.array(labels, dtype=np.int)

def get_wav_files(parent_dir,sub_dirs):
    wav_files = []
    for sub_dir in sub_dirs:
        for dirpath,dirnames,filenames in os.walk(parent_dir+sub_dir):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path = os.sep.join([dirpath, filename])
                    wav_files.append(filename_path)
    return wav_files

def padding(trest_features):
    xxx_data = []
    for mfccs in trest_features:
        while len(mfccs) < 173:  # 只要小于wav_max_len就补n_inputs个0
            mfccs.append([0] * 40)
        xxx_data.append(mfccs)
    return xxx_data

channel = grpc.insecure_channel(url)
grpc_stub = grpc_service_pb2_grpc.GRPCServiceStub(channel)

request = grpc_service_pb2.StatusRequest(model_name=model_name)
response = grpc_stub.Status(request)

input_name, output_names = parse_model(
    response, model_name)


def predict():
    wavFile = "./7061-6-0-0.wav"
    test = []
    test.append(wavFile)
    test_features, test_labels = extract_features(test)
    test_data = padding(test_features)
    test_data = np.array(test_data, dtype=np.float32)

    request = grpc_service_pb2.InferRequest()
    request.model_name = model_name
    request.model_version = -1

    request.meta_data.input.add(name=input_name, dims=[1, 173, 40])

    request.meta_data.batch_size = 1
    for output_name in output_names:
        output_message = api_pb2.InferRequestHeader.Output()
        output_message.name = output_name
        request.meta_data.output.extend([output_message])

    request.meta_data.output.add(name=output_name)

    request.raw_input.append(test_data.tobytes())

    resp = grpc_stub.Infer(request)

    prediction = np.frombuffer(resp.raw_output[0], dtype=np.float32).reshape([1, 10])
    result = np.argmax(prediction, 1)
    return result[0]

if __name__ == '__main__':
    result = predict()
    print ('result:', typeDict[result])
