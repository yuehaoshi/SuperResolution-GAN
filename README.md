## About
This is the repo for our final project which is an implementation of this [paper](https://arxiv.org/pdf/1609.04802.pdf) in Tensorflow2.

## How to run
To run the code you first have to download the dataset DIV2K

```
$ chmod a+x get_data.sh
$ ./get_data.sh
```

The model checkpoints and some outputs from SRREsnet are [here](https://drive.google.com/drive/folders/1tyo54tMsHTfhxE2aAkm8onYGbe2JfzlK?usp=sharing). For the checkpoints, just unzip the zipfile and place in the root folder.

### Training:
```
python3 training_*.py
```

### Inference:
```
python3 eval.py
```

### Dependencies:
```
tensorflow:2.8.0
tqdm
numpy
pillow

```



