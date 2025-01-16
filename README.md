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
To reproduce the results, run following commands:
```
python train.py --d_model 512 --gpu_ids 0
python test.py --plot_path xxx.mp4 --output_path xxx.pkl
```
If you have problem with limited gpu memory usage, try to decrease `--d_model` or use multi-gpu `--gpu_ids 0,1,2`.
- `--plot_path` make video of predicted playing movement. We here specify one of violinist for visualization.
- `--output_path` save predicted keypoints and ground truth, whose dimensions is N x K x C, where N is the number of frames, K is the number of keypoints and C is three axes x, y and z.

### Inference in the wild
If you want to make video and get predicted keypoints for custom audio data by pretrained model, run following commands:
```
python inference.py --inference_audio xxx.wav --plot_path xxx.mp4 --output_path xxx.pkl
```
`--plot_path` and `--output_path` are the same as described in **test.py**, and you need to put the path of your violin music on argument `--inference_audio`.
