# PEFT-JITCU
An Empirical Study on Parameter-Efficient Fine-Tuning for Just-In-Time Comment Updating

## Install

1. Clone this repository

```bash
git clone https://github.com/GalaxyLee4105/PEFT4JITCU.git
cd PEFT4JITCU
```

2. Install dependencies

```bash
conda create -n PEFT4JITCU python=3.10.8 -y
conda activate PEFT4JITCU
pip install -r requirement.txt
```

## Train

1. Data Preparation  

Please download this [ACL20](https://pan.baidu.com/s/1XYmfr7IOfWncH9wEPfDEGg?pwd=d84r) and And put it in the `dataset` folder.

```bash
cd Dataset
python data_processing.py
```

2. Start fine-tuning

```bash
cd ../Code/CodeGeeX2

# Fine-tuning with LORA techniques
cd LORA
bash train.sh 

# Fine-tuning with ADALoRA techniques
cd ADALORA
bash train.sh  

# Fine-tuning with IA3 techniques
cd IA3
bash train.sh     

# Fine-tuning with BITFIT techniques
cd BITFIT
bash train.sh   
```

3. Evaluation  

```bash
# Download required resources for evaluation
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
# METEOR relies on a Java environment
apt-get update && apt-get install -y default-jre
# GLEU relie on a py27 environment
conda create -n py27 python=2.7 
conda activate py27
conda install numpy=1.16 scipy=1.2
```
```bash
conda activate PEFT4JITCU

# Testing with LORA techniques
cd LORA
bash merge.sh     
bash test.sh

# Testing with ADALoRA techniques
cd ADALORA 
bash merge.sh     
bash test.sh

# Testing with IA3 techniques
cd IA3  
bash merge.sh     
bash test.sh

# Testing with BITFIT techniques
cd BITFIT   
bash merge.sh     
bash test.sh  
```

## Citation

```bibtex
@misc{Li2026empiricalstudyparameterefficientfinetuning,
      title={An Empirical Study on Parameter-Efficient Fine-Tuning for Just-In-Time Comment Updating}, 
      author={QunXing Li},
      year={2026},
}
```
