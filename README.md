# Music-to-playing-movement-generation-using-VQ-VAE-with-differential-motion-codebook
This is the pytorch implementation of Music-to-playing-movement-generation-using-VQ-VAE-with-differential-motion-codebook:  

## Quick start
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch and inference your own violin music.

### Dependencies
- Python 3+ distribution
- Pytorch >= 1.0.1, CUDA 10.0  
- Install requirements by running: `pip install -r requirement.txt`
- To visualize predictions, install ffmpeg by running: `apt-get install ffmpeg`

### Data

### Training from scratch
To reproduce the results, run the following commands:
```
python train.py --fps 30
python test.py --fps 30
```
You can specify 'fps' 30, 60, or 120 to train and test model at different resolutions.

### Inference in the wild
If you want to make video and get predicted keypoints for custom audio data by pretrained model, run the following commands:
```
python inference.py --fps 30 --input_audio xxx.wav --plot_path xxx.mp4 --output_path xxx.pkl
```
- `--fps` specify the model trained on different resolutions.
- `--input_audio` the path of your audio file.
- `--plot_path`  the path of plotted video.
- `--output_path` save predicted keypoints as pickle file, whose dimensions is N x K x C, where N is the number of frames, K is the number of keypoints and C is three axes x, y and z.
