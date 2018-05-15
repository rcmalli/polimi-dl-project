### Depth Estimation from Single Image ###
This project aims to create a model that predicts the depth map of a given rgb image. To fascilitate the implementation, this [template](https://github.com/benbenlijie/template-tensorflow-project) is deployed and edited.

## Dependencies 
- Python 3.6
- Tensorflow
- tqdm 
... TO BE LISTED

## Training 
- Training process can be runned with this command: 
	python train.py --config {CONFIGURATION_FILE.json}
For example,
	python train.py --config ./configs/NYUConfig.json

- By modifying the confugration file, different datasets can be used. 

## Testing 
- TODO

## Prediction
- Use this command for depth prediction:
	python predict.py --config ./configs/NYUConfig.json --input{IMAGE_PATH}
- Modify configuration file for model and output size.(Different model might output different sizes)

## Visualization in tensorboard
- Use this command for visualize loss,accuracy etc in tensorboard:
	tensorboard --logdir name:../experiments/NYU2/summary/train/
	
