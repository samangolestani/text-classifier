# Text Classifier

This project builds a simple text classification neural network to classify single words (jobs, fruits, vehicles, animals) using TensorFlow.

## Setup

```bash
pip install -r requirements.txt
```

## Steps

1. Generate dataset:
```bash
python scripts/create_dataset.py
```

2. Train the model:
```bash
python scripts/train_model.py
```

3. Predict new words:
```bash
python scripts/predict.py
```

## Project Structure
- `data/`: Holds generated dataset.
- `models/`: Holds trained model and tokenizers.
- `scripts/`: Scripts for creating data, training, predicting.
- `utils/`: Helper functions.
- `config.py`: Settings.

Enjoy!