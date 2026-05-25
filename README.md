#This work has been submitted to Applied Intelligence. The complete implementation will be made publicly available upon acceptance of the manuscript.
## Datasets Preparation

You can either directly download the preprocessed dataset directly from the links provided below or perform the preprocessing steps on your own.

### Preprocessed Datasets
| Dataset | Download |
| ------------------ | -------|
| S3DIS | [Download link](https://drive.google.com/file/d/1frJ8nf9XLK_fUBG4nrn8Hbslzn7914Ru/view?usp=drive_link) |
| ScanNet | [Download link](https://drive.google.com/file/d/19yESBZumU-VAIPrBr8aYPaw7UqPia4qH/view?usp=drive_link) |

### Preprocessing Instructions

**S3DIS**
1. **Download**: [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. **Preprocessing**: Re-organize raw data into `npy` files:
   ```bash
   cd preprocess
   python collect_s3dis_data.py --data_path [PATH_to_S3DIS_raw_data] --save_path [PATH_to_S3DIS_processed_data]
   ```
   The generated numpy files will be stored in `PATH_to_S3DIS_processed_data/scenes`.
3. **Splitting Rooms into Blocks**:
    ```bash
    python room2blocks.py --data_path [PATH_to_S3DIS_processed_data]/scenes
    ```


**ScanNet**
1. **Download**: [ScanNet V2](http://www.scan-net.org/).
2. **Preprocessing**: Re-organize raw data into `npy` files:
	```bash
	cd preprocess
	python collect_scannet_data.py --data_path [PATH_to_ScanNet_raw_data] --save_path [PATH_to_ScanNet_processed_data]
	```
   The generated numpy files will be stored in `PATH_to_ScanNet_processed_data/scenes`.
3. **Splitting Rooms into Blocks**:
    ```bash
    python room2blocks.py --data_path [PATH_to_ScanNet_processed_data]/scenes
    ```

After preprocessing the datasets, a folder named `blocks_bs1_s1` will be generated under `PATH_to_DATASET_processed_data`. Make sure to update the `data_root` entry in the .yaml config file to `[PATH_to_DATASET_processed_data]/blocks_bs1_s1/data`.

## Model weights
We provide the trained model weights across different few-shot settings and datasets below. 
The training and testing are using  RTX 3090 GPUs.
You could directly load these weights for evaluation or train your own models following the training instructions.
| Model name         | Dataset| CVFOLD | N-way K-shot | Model Weight |
| ------------------ | -------| ------|-----|----------------------------------- |
| **s30_1w1s**       | S3DIS  | 0 | 1-way 1-shot | [Download link](https://pan.baidu.com/s/1GL6K1_rI_MO2HlIS5tCi2A?pwd=8888)          |
| **s30_1w5s**       | S3DIS  | 0 | 1-way 5-shot |[Download link](https://pan.baidu.com/s/1YF2kfehzABFawcGY3XahwA?pwd=8888)     |
| **s30_2w1s**       | S3DIS| 0 | 2-way 1-shot | [Download link](https://pan.baidu.com/s/1pBl6ExrZK1Uy2Au9ZRcNSQ?pwd=8888)      |
| **s30_2w5s**       | S3DIS| 0 | 2-way 5-shot | [Download link](https://pan.baidu.com/s/1l_N2OsJWf847mCLj2_ayyw?pwd=8888)      |
| **s31_1w1s**       | S3DIS  | 1 | 1-way 1-shot | [Download link](https://pan.baidu.com/s/1JTWW-dIKqOFAU--jnPjsbw?pwd=8888)          |
| **s31_1w5s**       | S3DIS  | 1 | 1-way 5-shot |[Download link](https://pan.baidu.com/s/1BvsX_FI8BGJqN9affP1YVQ?pwd=8888)     |
| **s31_2w1s**       | S3DIS| 1 | 2-way 1-shot | [Download link](https://pan.baidu.com/s/16dpI_vmeACJkvZ8gBIclDQ?pwd=8888)      |
| **s31_2w5s**       | S3DIS| 1 | 2-way 5-shot | [Download link](https://pan.baidu.com/s/17r6Hp6tX7OlBRNvc23jY3w?pwd=8888)      |
| **sc0_1w1s**       | ScanNet  | 0 | 1-way 1-shot | [Download link](https://pan.baidu.com/s/16b3HbBh3LygSRPDYC34A6w?pwd=8888)          |
| **sc0_1w5s**       | ScanNet  | 0 | 1-way 5-shot |[Download link](https://pan.baidu.com/s/1cfIsa9U9knI4Q3SpGhL91w?pwd=8888)     |
| **sc0_2w1s**       | ScanNet| 0 | 2-way 1-shot | [Download link](https://pan.baidu.com/s/19BdHpifHVOEyfSP_dUcrWg?pwd=8888)      |
| **sc0_2w5s**       | ScanNet| 0 | 2-way 5-shot | [Download link](https://pan.baidu.com/s/1LPRjFfgFfkNUOyIY6LCz_w?pwd=8888)      |
| **sc1_1w1s**       | ScanNet  | 1 | 1-way 1-shot | [Download link](https://pan.baidu.com/s/1cZoNh1iFL23QpvuTVdb5ww?pwd=8888)          |
| **sc1_1w5s**       | ScanNet  | 1 | 1-way 5-shot |[Download link](https://pan.baidu.com/s/1HmMSiwi986kSmPIAZ5pA7g?pwd=8888)     |
| **sc1_2w1s**       | ScanNet| 1 | 2-way 1-shot | [Download link](https://pan.baidu.com/s/1aJC1gxrKOkdw5qo4XNKaoA?pwd=8888)      |
| **sc1_2w5s**       | ScanNet| 1 | 2-way 5-shot | [Download link](https://pan.baidu.com/s/1tA3lbLsbrohlC5kXfCaVlw?pwd=8888)      |

## Backbone pretraining
To begin, you will need to pretrain the backbone either on the S3DIS or ScanNet dataset. For consistency and ease of reproduction, we highly recommend using our pretrained backbone weights directly. You can find the pretrained weights and their corresponding download links below:

| Model name         | Dataset| CVFOLD|Model Weight                                               |
| ------------------ | -------| ------|---------------------------------------- |
| **s3_s1pre**       | S3DIS  | 1 |[Download link](https://drive.google.com/file/d/1zP2H3if2qEoXDe759WuCw9JJXeYbfHf2/view?usp=drive_link)          |
| **s3_s0pre**       | S3DIS  | 0 |[Download link](https://drive.google.com/file/d/1DyWv1PohFnGtX02lo2jcg09MEd7N9bVN/view?usp=drive_link)     |
| **sc_s1pre**       | ScanNet| 1 |[Download link](https://drive.google.com/file/d/1aeqEEckyo9v4ku5Bo6ixyXNm_Z2GmOQ9/view?usp=drive_link)      |
| **sc_s0pre**       | ScanNet| 0 |[Download link](https://drive.google.com/file/d/1RnWt3vshsrBrXaZ5Dlmz0RD6UQeReFwF/view?usp=drive_link)      |

Alternatively, you can perform the pretraining on your own. However, please note that doing so may result in more variability compared to the results reported in our paper.

To pretrain the backbone from scratch, run the following command, replacing `[PRETRAIN_CONFIG]` with the respective configuration file (`s3dis_stratified_pretraining.yaml` or `scannetv2_stratified_pretraining.yaml`), `[PATH_to_SAVE_BACKBONE]` with the desired path to save the backbone, and `[CVFOLD]` with either 0 or 1 depending on your few-shot setting:

```bash
python3 train_backbone.py --config config/[PRETRAIN_CONFIG] save_path [PATH_to_SAVE_BACKBONE] cvfold [CVFOLD]
```

## Environment

The following environment setup instructions have been tested on RTX 3090 GPUs with GCC 6.3.0.

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

If you encounter any issues with the above command, you can also install them individually:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch_points3d==1.3.0
pip install torch-scatter==2.1.1
pip install torch-points-kernels==0.6.10
pip install torch-geometric==1.7.2
pip install timm==0.9.2
pip install tensorboardX==2.6
pip install numpy==1.20.3
```


2. **Compile pointops**

Ensure you have `gcc`, `cuda`, and `nvcc` installed. Compile and install pointops2 as follows:

```bash
cd lib/pointops2
python setup.py install
```

## Testing

For testing, first modify `cvfold`, `n_way`, `k_shot`, and `num_episode_per_comb` in the corresponding config file, then run:

**S3DIS:**

```bash
python test_only.py --config config/s3dis_ablation.yaml --exp multiscale_dice_proto --weight [PATH_TO_CHECKPOINT] eval_split test
```

**ScanNetv2:**

```bash
python test_only.py --config config/scannetv2_ablation.yaml --exp multiscale_dice_proto --weight [PATH_TO_CHECKPOINT] eval_split test
```

The `--exp` flag selects the ablation variant. Available options: `baseline`, `multiscale_corr`, `dice_loss`, `proto_noise`, `dice_multiscale`, `dice_proto_noise`, `multiscale_proto`, `multiscale_dice_proto`.

For visualization (S3DIS only), add `forvis 1`:

```bash
python test_only.py --config config/s3dis_ablation.yaml --exp multiscale_dice_proto --weight [PATH_TO_CHECKPOINT] eval_split test forvis 1
```

> **Note:** It is common to observe fluctuations in mIoU by approximately 1.0%. This variability may be attributed to the relatively small size of the training set. The variance in performance on ScanNetv2 tends to be smaller compared to S3DIS due to its larger size.

## Acknowledgement

Our codebase is built upon [COSeg](https://github.com/ZhaochongAn/COSeg). The dataset preprocessing scripts, preprocessed dataset links, environment setup instructions, and CUDA point cloud operations (`lib/pointops`, `lib/pointops2`) are adopted from the original COSeg repository. The backbone network is based on [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer). We thank the authors for making their code publicly available.

## Citation

If you find this project useful, please consider citing the original COSeg work:

```bibtex
@inproceedings{an2024rethinking,
  title={Rethinking Few-shot 3D Point Cloud Semantic Segmentation},
  author={An, Zhaochong and Sun, Guolei and Liu, Yun and Liu, Fayao and Wu, Zongwei and Wang, Dan and Van Gool, Luc and Belongie, Serge},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
