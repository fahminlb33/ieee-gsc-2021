# Challenge 1 - Application Failure Prediction

## Requirements

You'll need the virtual environment from `challenge1` folder to run the `transform_dataset.py` script. However, to run the script in the `modelling_sparknlp.ipynb` and `modelling_tensorflow.ipynb` you'll need to upload the notebook into Kaggle Notebooks. This is because I use Spark NLP library with a preconfigured install script for Kaggle. Also, I haven't tried running the script on my local computer. The use of Kaggle Notebook is becuase I don't have a computer with sufficent processing power to run the script.

## How to run

There are two type of script used here, one is used to transform the dataset before the modelling process and the actual script to perform the modelling process.

## Python files

The file `transform_dataset.py` is used to transform the provided training data to a suitable format for the training process. This is necessary because I use two different library to perform the modelling task, TensorFlow and Spark NLP.

```bash
> cwd
/home/ieee-gsc-2021/challenge2

> python transform_dataset.py dataset/training.csv dataset/tf.csv --mode tensorflow
Processed 5000 lines.
  ID                                              Tweet  0  1  2  3  4  5  6  7  8  9  10
0  1  NO JOKE I WILL HOP ON A PLANE RN! (Well after ...  1  0  0  0  0  0  0  0  0  0   1
1  2  BanMediaHouse whose is responsible for spreadi...  0  0  0  0  0  0  1  0  0  0   0
2  3  Im waiting for someone to say to me that all t...  0  0  0  1  1  0  0  0  0  0   0
3  4  He is a liar. Proven day night. Time again. Li...  0  0  0  0  0  0  1  0  0  0   0
4  5  NEW: U.S. CoronaVirus death toll reaches 4,000...  0  0  0  0  0  0  0  0  1  0   0

> python transform_dataset.py dataset/training.csv dataset/spark.parquet --mode sparknlp
Processed 5000 lines.
  id                                               text                  labels
0  1  NO JOKE I WILL HOP ON A PLANE RN! (Well after ...    [optimistic, joking]
1  2  BanMediaHouse whose is responsible for spreadi...               [annoyed]
2  3  Im waiting for someone to say to me that all t...  [pessimistic, anxious]
3  4  He is a liar. Proven day night. Time again. Li...               [annoyed]
4  5  NEW: U.S. CoronaVirus death toll reaches 4,000...              [surprise]
```

But you don't actually need to run this script since the script in the Jupyter Notebook already has this code. This code is used when I tried to run the modelling process in my computer, but didn't succeeded.

### Jupyter Notebook files

1. Create a new notebook in the problem page by clicking **New Notebook**
2. Click File > Upload notebook
3. Drag and drop the IPYNB file to upload
4. Click **Run All** to execute all the cells and produce an output predictions file.

For the TensorFlow model, you can choose between a variety of different pretrained models by changing the URL used in the cell.
