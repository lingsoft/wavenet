## Train a vocoder with finnish [audio](https://github.com/alastaa/tacotron2-samples/tree/master/audio)
First clone the project and copy audio directory into wavenet_vocoder directory.

## Install dependencies:
The project was tested with python 3.7.11 
### Install packages
````
python -m venv venv && source venv/bin/activate
pip install -r req.txt
````

## Spliting the data into train, dev, eval
````
 sh ./egs/gaussian/run.sh --stage 0 --stop-stage 0
````
## Feature extraction (MelSpectral) from the data, this will generate dump directory with original and normalized mel spectrogram data
````
 sh ./egs/gaussian/run.sh --stage 1 --stop-stage 1
````
## Start training the model with default configuration under gaussian recipe (json)[egs/gaussian/conf/gaussian_wavenet.json]
````
 sh CUDA_VISIBLE_DEVICES="0,1" ./egs/gaussian/run.sh --stage 2 --stop-stage 2
````
## Synthesize audio files using the trained model given mel spectrogram from dump/norm/eval/*.npy input files.
````
 sh ./egs/gaussian/run.sh --stage 3 --stop-stage 3
````
The audio generated file will be under: wavenet_vocoder/exp/lj_train_no_dev_gaussian_wavenet/generated/checkpoint_latest/eval/
