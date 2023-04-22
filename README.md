# 🏆 Trear: Transformer-based RGB-D Egocentric Action Recognition

<img src="https://user-images.githubusercontent.com/87626122/217718225-9222f201-0f80-4458-8867-43d3deb0f5ae.png" width="1000" height="350">

### Trear: Transformer-based RGB-D Egocentric Action Recognition
### Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, and Wanqing Li
### Abstract: In this paper, we propose a Transformer-based RGB-D egocentric action recognition framework, called Trear. It consists of two modules, inter-frame attention encoder and mutual-attentional fusion block. Instead of using optical flow or recurrent units, we adopt self-attention mechanism to model the temporal structure of the data from different modalities. Input frames are cropped randomly to mitigate the effect of the data redundancy. Features from each modality are interacted through the proposed fusion block and combined through a simple yet effective fusion operation to produce a joint RGB-D representation. Empirical experiments on two large egocentric RGB-D datasets, THU-READ and FPHA, and one small dataset, WCVS, have shown that the proposed method outperforms the state-of-the-art results by a large margin.

# 📝 Installation
#### For all methods decribed in the paper, it requires to have:
* Python 3.7
* PyTorch 1.10
* opencv-python 4.5.1.48
* Pillow 8.3.1

# 🔔  Data preparation
#### Download FPHA(First-Person Hand Action Benchmark) and locate it in `data` folder.
#### Refer to the `FPHA_dataset.txt` in that folder.
#### Also annotation files are in `annotations` folder and there are five files.
#### We need only three files to train/test.
* FPHA_train_list.txt
* FPHA_val_list.txt
* FPHA_test_list.txt

`FPHA_train.txt`, `FPHA_test.txt` are only used to make above files.

#### Annotation files are composed of three elements of each action folder.

`Directory of action folder` `Total numder of frames of in that folder` `label numder(0~44)`

# 💻  Train/Test
#### To train a model, use the `main.py` script.

```
python main.py FPHA Motion ./annotations/FPHA_train_list.txt  --arch resnet50 --num_segment 3 --lr 0.0001 --lr_steps 30 60 --epochs 50 -b 4 --snapshot_pref train --val_list ./annotations/FPHA_test_list.txt --gpus 0
```

#### You can see the final test result at the end of the epoch like below form.
`Test Results: rgb Prec@5 96.522 Loss 3.80745        depth Prec@5 97.391 Loss 3.80802        fus Prec@5 97.565 Loss 3.81364`
# Citation
```
@InProceedings{
    author    = {Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, and Wanqing Li},
    title     = {Trear: Transformer-based RGB-D Egocentric Action Recognition},
    booktitle = {},
    month     = {January},
    year      = {2021},
    pages     = {}
}
```


