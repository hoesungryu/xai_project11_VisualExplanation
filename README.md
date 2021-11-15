# xai_project11_VisualExplanation

## Dataset
[A Large-Scale Dataset for Segmentation and Classification](https://www.kaggle.com/crowww/a-large-scale-fish-dataset)
```
wget --load-cookies ./cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ./cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uLMes0_V04yOttZVxPQo-BF_51pyK8Kz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uLMes0_V04yOttZVxPQo-BF_51pyK8Kz" -O Fish_Dataset.zip && rm -rf ./cookies.txt
```
## Train
```
python train.py --data_dir /path/your/data/dir --save_dir /path/your/save/dir
```
## Evaluation
```
python evaluation.py --image_dir /path/your/image/dir --label_dir /path/your/label/dir --target_class 0~8
```
Class\
0: Black Sea Sprat,\
1: Gilt-Head Bream\
2: Hourse Mackerel\
3: Red Mullet\
4: Red Sea Bream\
5: Sea Bass\
6: Shrimp\
7: Striped Red Mullet\
8: Trout

## Requirment
```
pip install -r requirements.txt
```
torch                  1.7.1\
torchcam               0.2.0\
torchvision            0.8.2\
tqdm                   4.62.3\
Pillow                 8.1.1\
opencv-python          3.4.8.29\
numpy                  1.21.2\
matplotlib             3.3.2
