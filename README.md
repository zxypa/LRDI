# Structured Feature Interaction with Stability Constraints for Image Super-Resolution

## Environment

- Python 3.9
- PyTorch 2.0.1
- CUDA 11.x
- Ubuntu 18.04 / 20.04 (recommended)

---

## Training

### Data Preparation

- Download the training dataset **DF2K**  
  ([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) + [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar))  
  and place them into the folder `./datasets`.

- For faster data reading and preprocessing, it is recommended to follow the dataset preparation guide from  
  [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md).

The directory structure should be:
```

datasets/
├── DIV2K/
└── Flickr2K/

````

### Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
  --use-env --nproc_per_node=8 --master_port=1145 \
  basicsr/train.py -opt options/train/SFIS_SRx4.yml --launcher pytorch
````

---

## Testing

### Data Preparation

* Download the benchmark datasets:
  **Set5**, **Set14**, **BSD100**, **Urban100**, **Manga109**
  [[download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]

* Place the testing datasets into the folder `./datasets`.

The directory structure should be:

```
datasets/
 ├── Set5/
 ├── Set14/
 ├── BSD100/
 ├── Urban100/
 └── Manga109/
```

### Testing Command

```bash
python basicsr/test.py -opt options/test/SFIS_SRx4.yml
```

The testing results (reconstructed images and PSNR / SSIM values) will be saved automatically.

---


```

---

如果你愿意，我可以下一步直接帮你补一个**“Method Overview + Network Architecture 图说明版 README”**，或者帮你把 **SFIS 和论文摘要完全对齐**，审稿人会非常舒服。
```
