# YOLOv9 for Fracture Detection

>[YOLOv9 for Fracture Detection in Pediatric Wrist Trauma X-ray Images](https://arxiv.org/abs/2403.11249)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolov9-for-fracture-detection-in-pediatric/object-detection-on-grazpedwri-dx)](https://paperswithcode.com/sota/object-detection-on-grazpedwri-dx?p=yolov9-for-fracture-detection-in-pediatric)

## Comparison
<p align="left">
  <img src="img/figure_comparison.jpg" width="480" title="details">
</p>

## Performance
| Model | Test Size | Param. | FLOPs | F1 Score | AP<sub>50</sub><sup>val</sup> | AP<sub>50-95</sub><sup>val</sup> | Speed |
| :--: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| YOLOv8 | 640 | 43.61M | 164.9G | 0.59 | 62.44% | 40.32% | 3.6ms |
| YOLOv8+SA | 640 | 43.64M | 165.4G | 0.62 | 63.99% | 41.49% | 3.9ms |
| YOLOv8+ECA | 640 | 43.64M | 165.5G | 0.61 | 62.64% | 40.21% | 3.6ms |
| YOLOv8+GAM | 640 | 49.29M | 183.5G | 0.60 | 63.32% | 40.74% | 8.7ms |
| YOLOv8+ResGAM | 640 | 49.29M | 183.5G | 0.62 | 63.97% | 41.18% | 9.4ms |
| YOLOv8+ResCBAM | 640 | 53.87M | 196.2G | 0.62 | 62.95% | 40.10% | 4.1ms |
| **YOLOv9-C** | 640 | **51.02M** | **239.0G** | **0.64** | **65.31%** | **42.66%** | **5.2ms** |
| **YOLOv9-E** | 640 | **69.42M** | **244.9G** | **0.64** | **65.46%** | **43.32%** | **6.4ms** |

## Citation
If you find our paper useful in your research, please consider citing:

    @article{chien2024yolov9,
      title={YOLOv9 for fracture detection in pediatric wrist trauma X-ray images},
      author={Chien, Chun-Tse and Ju, Rui-Yang and Chou, Kuang-Yi and Chiang, Jen-Shiun},
      journal={Electronics Letters},
      volume={60},
      number={11},
      pages={e13248},
      year={2024},
      publisher={Wiley Online Library}
    }
    
## Requirements
* Linux (Ubuntu)
* Python = 3.9
* Pytorch = 1.13.1
* NVIDIA GPU + CUDA CuDNN

## Environment
```
  pip install -r requirements.txt
```

## Overall Flowchart
<p align="left">
  <img src="img/figure_flowchart.jpg" width="1024" title="details">
</p>

## Dataset Split
* GRAZPEDWRI-DX Dataset [(Download Link)](https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193)
* Download dataset and put images and annotatation into `./GRAZPEDWRI-DX_dataset/data/images`, `./GRAZPEDWRI-DX_dataset/data/labels`.
  ```
    python split.py
  ```
* The dataset is divided into training, validation, and testing set (70-20-10 %) according to the key `patient_id` stored in `dataset.csv`.
* The script then will move the files into the relative folder as it is represented here below.


       GRAZPEDWRI-DX_dataset
          └── data   
               ├── images
               │    ├── train
               │    │    ├── train_img1.png
               │    │    └── ...
               │    ├── valid
               │    │    ├── valid_img1.png
               │    │    └── ...
               │    └── test
               │         ├── test_img1.png
               │         └── ...
               └── labels
                    ├── train
                    │    ├── train_annotation1.txt
                    │    └── ...
                    ├── valid
                    │    ├── valid_annotation1.txt
                    │    └── ...
                    └── test
                         ├── test_annotation1.txt
                         └── ...


The script will create 3 files: `train_data.csv`, `valid_data.csv`, and `test_data.csv` with the same structure of `dataset.csv`.

## Data Augmentation
* Data augmentation of the training set using the addWeighted function doubles the size of the training set.
```
  python imgaug.py --input_img /path/to/input/train/ --output_img /path/to/output/train/ --input_label /path/to/input/labels/ --output_label /path/to/output/labels/
```
For example:
```
  python imgaug.py --input_img ./GRAZPEDWRI-DX/data/images/train/ --output_img ./GRAZPEDWRI-DX/data/images/train_aug/ --input_label ./GRAZPEDWRI-DX/data/labels/train/ --output_label ./GRAZPEDWRI-DX/data/labels/train_aug/
```

* The path of the processed file is shown below:

       GRAZPEDWRI-DX_dataset
          └── data   
               ├── images
               │    ├── train
               │    │    ├── train_img1.png
               │    │    └── ...
               │    ├── train_aug
               │    │    ├── train_aug_img1.png
               │    │    └── ...
               │    ├── valid
               │    │    ├── valid_img1.png
               │    │    └── ...
               │    └── test
               │         ├── test_img1.png
               │         └── ...
               └── labels
                    ├── train
                    │    ├── train_annotation1.txt
                    │    └── ...
                    ├── train_aug
                    │    ├── train_aug_annotation1.txt
                    │    └── ...
                    ├── valid
                    │    ├── valid_annotation1.txt
                    │    └── ...
                    └── test
                         ├── test_annotation1.txt
                         └── ...
                         
## Weights
If you plan to use pretrained models to train, you need put them into `./weights/`.
* You can get the YOLOv9 pretained models on the MS COCO 2017 Dataset through [YOLOv9 official GitHub](https://github.com/WongKinYiu/yolov9).
* Use gdown to download the trained model from our GitHub:
```
  gdown https://github.com/RuiyangJu/YOLOv9-Fracture-Detection/releases/download/Trained/weights.zip
```

## Train & Validate
Before training the model, make sure the path to the data in the `./data/meta.yaml` file is correct.

* meta.yaml
```
  # patch: /path/to/GRAZPEDWRI-DX/data
  path: 'E:/GRAZPEDWRI-DX/data'
  train: 'images/train_aug'
  val: 'images/valid'
  test: 'images/test'
```

* Arguments

| Key | Value | Description |
| :---: | :---: | :---: |
| workers | 8 | number of worker threads for data loading (per RANK if DDP) |
| device | None | device to run on, i.e. device=0,1,2,3 or device=cpu |
| model | None | path to model file, i.e. yolov8n.pt, yolov8n.yaml |
| batch | 16 | number of images per batch (-1 for AutoBatch) |
| data | None | path to data file, i.e. coco128.yaml |
| img | 640 | size of input images as integer, i.e. 640, 1024 |
| cfg | yolo.yaml | path to model.yaml, i.e. yolov9-c.yaml |
| weights | None | initial weights path |
| name | exp | save to project/name |
| hyp | data/hyps/hyp.scratch-high.yaml | hyperparameters path |
| epochs | 100 | number of epochs to train for |

* Example
```
  python train_dual.py --workers 8 --device 0 --batch 16 --data data/meta.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights weights/yolov9-c.pt --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 100 --close-mosaic 15
```

## Related Works

<details><summary> <b>Expand</b> </summary>

* [https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8](https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8)
* [https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8](https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8)
* [https://github.com/RuiyangJu/YOLOv8_Global_Context_Fracture_Detection](https://github.com/RuiyangJu/YOLOv8_Global_Context_Fracture_Detection)

</details>
