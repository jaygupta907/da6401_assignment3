# 🧠 DA6401 Assignment: Transliteration Using seq2seq modelling

## 🔧 1️⃣ Setup Instructions
- Ensure you have Conda installed.
- Create an environment and install dependencies.
```bash
conda env create -n Sequence python=3.10
conda activate Sequence 
pip install -r requirements.txt
```

## 📥 Download Dataset 
```bash
python download.py
```

## 🏋️ Training
### Train a vanilla Seq2Seq model:
```bash
python train.py --attention False
```
### Train a Seq2Seq model with attention:
```bash
python train.py --attention True
```

## 🧪 Evaluation
### Evaluate a trained model:
```bash
python test.py --attention True
```

## 🔍 Visualizations

### 🔥 Attention Heatmap
```bash
python plot_heatmap.py --attention True
```

### 🧬 Connectivity Graph & Interactive UI
Download the weights and vocab from the link provided in the PDF and store them in saved folder in main directoty.
Launch the FastAPI web app to explore word-level attention and hidden state similarities:
```bash
python visualise.py
```
Then visit http://localhost:8000