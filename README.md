### Set up 
Please follow the instructions (prerequisites) in our the MS Transformer repository: https://github.com/yolish/multi-scene-pose-transformer 

### Usage

The entry point for training and testing is the main.py script in the root directory

  For detailed explanation of the options run:
  ```
  python main.py -h
  ```
  
  For example, in order to train our model on the 7Scenes dataset run: 
  ```
    python main.py 
    c2f-ems-transposenet 
    train 
    models/backbones/efficient-net-b0.pth 
    /path/to/7scenes-datasets .
    /datasets/7Scenes/7scenes_all_scenes.csv 
    7scenes_config.json 
    --checkpoint_path path/to/pretrained-ms-transformer
  ```
  Your checkpoints (.pth file saved based on the number you specify in the configuration file) and log file
  will be saved under an 'out' folder.
  
  To run on cambridge, you will need to change the configuration file to ```CambridgeLandmarks_config.json``` for initial training and ```CambridgeLandmarks_finetune_config.json``` for fine-tuning (see details in our paper). 
  
  In order to test your model, for example on the ShopFacade scene from the CambridgeLandmarks dataset:
  ```
    python main.py 
    c2f-ems-transposenet
    test
    models/backbones/efficient-net-b0.pth
    path/to/CambridgeLandmarks/
    ./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_ShopFacade_test.csv
    CambridgeLandmarks_config.json
    --checkpoint_path
    path/to/checkpoint
    --cluster_predictor_position
    ./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_ShopFacade_position_4_classes.sav
    --cluster_predictor_orientation
    ./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_ShopFacade_orientation_4_classes.sav
  ```

  
  
  
