# AI6126 CelebAMask Face Parsing
This repository contains my work for the CelebAMask Face Parsing project as part of NTU AI6126 Advanced Computer Vision class.

The goal of this mini challenge is to design and train a face parsing network using the CelebAMask-HQ Dataset. A mini-dataset consisting of 1000 training and 100 validation pairs of
images is used, where both images and annotations have a resolution of 512 x 512.

The performance of the network will be evaluated based on the F-measure between the
predicted masks and the ground truth of the test set.

### Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/Ardacandra/ai6126_CelebAMask_face_parsing.git
cd ai6126_CelebAMask_face_parsing
```

2. Set up the conda environment

```bash
conda create -n ai6126_CelebAMask_face_parsing python=3.10 -y
conda activate ai6126_CelebAMask_face_parsing
```

3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

4. Prepare the CelebAMask-HQ Dataset

Download the dataset and store in `./data` folder. The format should be as follows:

```
data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    └── images/
```

Try running the visualization script to make sure the dataset is stored correctly:

```bash
python visualize_samples.py
```

### Training and Evaluation

1. Set `output.run_id`, `model.name`, and other training hyperparameters in `config.yaml`.
2. Train:

```bash
python src/train.py
```

3. Generate validation predictions:

```bash
python src/evaluate.py
```

4. Prepare submission package (matches `data/sample-submission/masks` format):

```bash
python prepare_submission.py
```

This creates:

- `out/<run_id>/submission/masks/*.png`
- `out/<run_id>/<run_id>_submission.zip`