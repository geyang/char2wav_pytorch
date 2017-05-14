# Deep Machine Translation [Work-In-Progress 2017-04-02]
[![](https://img.shields.io/badge/link_on-GitHub-brightgreen.svg?style=flat-square)](https://github.com/episodeyang/deep_machine_translation/tree/master#deep-machine-translation-work-in-progress-2017-04-02)


This is a fun week-long project I did to implement a sequence to sequence model in PyTorch. The project uses language pairs from the Anki project as the training set.

## Usage

#### work-in-progress

1. First unzip a language pair. use `eng-cmn.txt` from the [training-data](training-data/) folder for example.

2. Then run this script bellow (will be put into the make file):

    ```bash
    python traing.py python train.py -cf=checkpoints
    ```

### Command Line Options

```bash
usage: train.py [-h] [-d DEBUG] [-cf CHECKPOINT_FOLDER] [-cp CHECKPOINT]
                [--checkpoint-batch-stamp CHECKPOINT_BATCH_STAMP]
                [-il INPUT_LANG] [-ol OUTPUT_LANG]
                [--max-data-len MAX_DATA_LEN] [--dash-id DASH_ID]
                [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
                [--n-epoch N_EPOCH] [-e EVAL_INTERVAL]
                [--teacher-forcing-r TEACHER_FORCING_R] [-s SAVE_INTERVAL]
                [--n-layers N_LAYERS] [--bi-directional BI_DIRECTIONAL]

Sequence-To-Sequence Model in PyTorch

optional arguments:
  -h, --help            show this help message and exit
  -d DEBUG, --debug DEBUG
                        debug mode prints more info
  -cf CHECKPOINT_FOLDER, --checkpoint-folder CHECKPOINT_FOLDER
                        folder where it saves checkpoint files
  -cp CHECKPOINT, --checkpoint CHECKPOINT
                        the checkpoint to load
  --checkpoint-batch-stamp CHECKPOINT_BATCH_STAMP
                        the checkpoint to load
  -il INPUT_LANG, --input-lang INPUT_LANG
                        code name for the input language
  -ol OUTPUT_LANG, --output-lang OUTPUT_LANG
                        code name for the output language
  --max-data-len MAX_DATA_LEN
                        maximum length for input output pairs (words)
  --dash-id DASH_ID     maximum length for input output pairs
  --batch-size BATCH_SIZE
                        maximum length for input output pairs
  --learning-rate LEARNING_RATE
                        maximum length for input output pairs
  --n-epoch N_EPOCH     number of epochs to train
  -e EVAL_INTERVAL, --eval-interval EVAL_INTERVAL
                        evaluate model on validation set
  --teacher-forcing-r TEACHER_FORCING_R
                        Float for the teacher-forcing ratio
  -s SAVE_INTERVAL, --save-interval SAVE_INTERVAL
                        evaluate model on validation set
  --n-layers N_LAYERS   maximum length for input output pairs
  --bi-directional BI_DIRECTIONAL
                        whether use bi-directional module for the model
```

## Key Learning

- **Good Demo train fast**. Sean Roberson's demos are very nice for teaching, partially because the training converges very quickly. Most of his demo converge in less than half an hour on a MBP. 

- **Write Evaluation Functions Early**. You can't manually check if the results make sense until you have writen the evaluation functions. 
    
    When everything is done correctly, the evaluation gives sensible results very quickly.

- **Teacher forcing can be a hyper parameter**. During training, we can tune the teacher forcing ratio between 0 and 1.

- **Mini-batch hugely improves training speed**. Here I used a mini-batch of 128 pairs. For the small English-to-Chinese anki dataset, the training takes about 1 min 30 seconds on an i7 PC. And the translated result is acceptable within a few epochs.

- **Training and loss function need more work**. The loss function used here feels a bit unsatisfactory.

## TODO List

- [x] polish demo
- [x] write sequence to sequence example
- [x] get both training and evaluation to work
- [ ] Add BLUE as accuracy metric
- [ ] Add confusion matrix in demo.
- [ ] Add unzip script for languages
- [ ] polish repo
- [ ] Compare results with attention model

## DONE
- [x] write data scraper, download zip files from anki
- [x] convert zip to text file
- [x] write evaluation function
