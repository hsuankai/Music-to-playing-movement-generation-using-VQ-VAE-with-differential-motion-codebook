# Music-to-playing-movement-generation-using-VQ-VAE-with-differential-motion-codebook
This repository provides a PyTorch implementation of Music-to-Playing-Movement-Generation-using-VQ-VAE-with-Differential-Motion-Codebook. The codebase enables training and inference for generating violin playing movement from audio input.

## Quick start
To get started as quickly as possible, follow the instructions in this section. This should allow you to train a model from scratch and perform inference on your own violin music.

### Dependencies
- Python 3+ distribution
- Pytorch >= 1.0.1, CUDA 10.0  
- Install requirements by running: `pip install -r requirement.txt`
- To visualize predictions, install ffmpeg by running: `apt-get install ffmpeg`

### Data

### Training from scratch
To reproduce the results, run the following commands:
```
python train.py --fps 30 --model motionvqvae
python test.py --fps 30 --model motionvqvae
```
- Specify `--model` to train either the motionvqvae or audio2motion model.
* Specify --fps as 30, 60, or 120 to train and test the model at different resolutions.

### Inference in the wild
If you want to make a video and get predicted keypoints for custom audio data using a pretrained model, run the following command:
```
python inference.py --fps 30 --audio xxx.wav --plot_path results/animation --output_path results/keypoints
```
- `--fps` Specify the frame rate that matches the model you trained (30, 60, or 120).
- `--audio` Path to your audio file (e.g., xxx.wav).
- `--plot_path`  Path to save the generated animation video.
- `--output_path` Path to save predicted keypoints as a pickle file (dimensions: N x K x C, where N = number of frames, K = number of keypoints, C = x, y, z axes).
