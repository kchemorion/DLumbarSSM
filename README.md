# MultiView-SpineNet 
A Novel Architecture for Multi-view Spine Analysis with Cross-attention and Anatomical Constraints


## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize data:
- Place DICOM images in `data/raw/train_images/`
- Place CSV files in `data/annotations/`

4. Directory Structure:
```
spine_pipeline/
├── data/
│   ├── raw/
│   │   └── train_images/
│   ├── preprocessed/
│   └── annotations/
│       ├── train.csv
│       ├── train_label_coordinates.csv
│       └── train_series_descriptions.csv
├── src/
│   ├── preprocessing/
│   ├── segmentation/
│   ├── shape_modeling/
│   └── visualization/
├── models/
└── results/
```

## Usage

1. Run initial analysis:
```bash
python src/main.py
```

2. Results will be saved in:
- `results/statistics/`: Dataset statistics and visualizations
- `results/processing_results/`: Processing results for individual cases