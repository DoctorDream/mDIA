##What can crawl do

1. Zip: Compress the original reddit comment data (extract some tags) and identify the language, and then store it in the detaileddata folder
2. Generate: Using reddit API to obtain parent comments, identify languages and build dialogue



### Prepare the raw data

from  [this website](https://files.pushshift.io/reddit/comments/) download the data and put it in the RawData folder, and then unzip it.

### Installation

1. First, you need to install the corresponding pytorch according to the CUDA version.
2. Install dependent packages

```bash
$ pip install -r requirements.txt
```
Python 3.8+ is recommended.

### Zip

cd ./crawl and then run the command below:
   ```python
   python main.py zip RC_2020-02
   ```
   Replace `RC_2020-02` with file you need to zip

### Generate

After you complete the step `zip`，cd ./crawl and then run the command below
   ```python
   python main.py generate RC_2020-02
   ```
Replace `RC_2020-02` with file you need to generate conversations

## What can train_val do

1. Train: Use the collected corpus to fine tune the model to the dialogue generation task
2. Eval: Evaluate the generation effect of the model in various languages

### Train

cd ./train_val and then run the command below:
```python
   python main.py --do train --lang ko
```

### Evaluate

cd ./train_val and then run the command below:
```python
   python main.py  --model_name microsoft/DialoGPT-large --do evaluate --lang ko
```

The training and evaluate script accept several arguments to tweak the training:

| Argument   | Type | Default value   | Description                                       |
| ---------- | ---- | --------------- | ------------------------------------------------- |
| model_name | str  | google/mt5-base | which model to train or evaluate                  |
| do         | str  | train           | to train or to eval                               |
| lang       | str  |                 | which language to train or evaluate               |
| translated | bool | False           | whether to use MarianMT                           |
| raw        | bool | False           | whether to use zero-shot DialoGPT                 |
| small      | int  | 10000           | desice how many data to used to train or evaluate |
| gpu        | str  | 0               | which gpu to use                                  |