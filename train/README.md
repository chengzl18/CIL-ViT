# carla_cil_vit_training


The transformer implementation to the conditional imitation learning policy in "End-to-end Driving: A Survey and An Implementation".

## Requirements
python 3.6
pytorch == 1.10.1       
transformers == 4.18.0       
opencv    
imagaug    
h5py    

## Train
**train-dir** and **eval-dir** should point to where the [Carla dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) located.
Dataset seperation details can be seen at [paper](https://ram-lab.com/file/tailei/vr_goggles/index.html).
```
To run the baseline CNN model:
$ python main.py --batch-size 1000 --workers 16 -- model_type 'CNN'
    --train-dir "path/to/AgentHuman/SeqTrain/"
    --eval-dir "path/to/AgentHuman/SeqVal/"
    --gpu 0
    --id training_CNN
    
To run our ViT implementation:
$ bash run.sh

To run our MAE implementation:
$ bash run_MAE.sh

To run our DETR implementation:
$ bash run_DETR.sh
```

## Dataset
Please check the original [dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) of Carla Imitation Learning.    
Please check this [issue](https://github.com/carla-simulator/imitation-learning/issues/1) for data augmentation.

## Reference

[onlytailei/carla_cil_pytorch](https://github.com/onlytailei/carla_cil_pytorch/tree/master)    
[carla-simulator/imitation-learning](https://github.com/carla-simulator/imitation-learning)    
[mvpcom/carlaILTrainer](https://github.com/mvpcom/carlaILTrainer)    
[End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)    
[CARLA: An Open Urban Driving Simulator](http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)    
[VR-Goggles for Robots: Real-to-sim Domain Adaptation for Visual Control](https://ram-lab.com/file/tailei/vr_goggles/index.html)
