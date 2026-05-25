#The relevant code files will be made publicly available upon acceptance of the manuscript. Below are our training logs and pre-trained model weights for your reference.

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

For incompatible installation issues, such as wanting a higher torch version (e.g., 2.1.0) but conflicts with `torch_points3d`, please refer to this thread: https://github.com/ZhaochongAn/COSeg/issues/16 or feel free to open a new discussion for further assistance.

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
