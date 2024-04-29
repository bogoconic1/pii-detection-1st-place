# Kaggle PII Data Detection Competition - 1st Place Solution Full Code

This repository contains the code and configurations for our winning solution in the **PII Data Detection** competition hosted by The Learning Agency Lab. Our team's approach and results are detailed in the [competition discussion on Kaggle](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374).

## Inferencing

The inference process is detailed in the notebook available at [Kaggle: PII 1st Place Solution](https://www.kaggle.com/code/yeoyunsianggeremie/pii-1st-place-solution). It outlines the procedure for using the trained models to predict and process test data.

## Table of Contents

- [Kaggle PII Data Detection Competition - 1st Place Solution Full Code](#kaggle-pii-data-detection-competition---1st-place-solution-full-code)
  - [Inferencing](#inferencing)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [Hardware](#hardware)
    - [Software](#software)
    - [Dependencies](#dependencies)
    - [Datasets](#datasets)
  - [Training](#training)

## Setup

### Hardware

- **Instance**: Ubuntu 20.04.5 LTS (128 GB boot disk)
- **CPU**: Intel(R) Xeon(R) Silver 4216 @ 2.10GHz (7 vCPUs)
- **GPU**: 8 x NVIDIA A100 40GB

### Software

- **Python**: 3.10.13
- **CUDA**: 12.1

### Dependencies

Clone the repository and install the required Python packages:

```shell
git clone https://github.com/bogoconic1/pii-detection-1st-place.git
cd pii-detection-1st-place
pip install -r requirements.txt
```

### Datasets

Ensure the Kaggle API is installed and set up. Use the following script to download the necessary datasets:

```shell
sh ./setup_datasets.sh
```

Note: The script creates a `data` folder in the parent directory and downloads external datasets there.

## Training

Our solution involves five Deberta-v3-large models, incorporating different architectures for diversity and performance. Below are some variants and their training commands:

- Multi-Sample Dropout Custom Model: Improves training stability and performance.

    ```shell
    python train_multi_dropouts.py
    ```

- BiLSTM Layer Custom Model: Adds a BiLSTM layer to enhance feature extraction, includes specific initialization to prevent NaN loss issues.

    ```shell
    python train_bilstm.py
    ```

- Knowledge Distillation: Utilizes well-performing models as teachers to boost a student model's performance, leveraging disparate datasets. It requires a teacher model. We used the best of Multi-Sample dropout models.
Note: it requires a teacher model to be distlled with. We used the best of multi-sample dropout models.

    ```shell
    python train_distil.py
    ```

- Experiment 073: Uses augmented data with name swaps.

    ```shell
    python train_exp073.py
    ```

- Experiment 076: Introduces a random addition of consequential names to training data.

    ```shell
    python train_exp076.py
    ```
