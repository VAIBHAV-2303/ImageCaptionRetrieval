# ImageCaptionRetrieval

## Description

Pytorch code for training an image-caption retrieval model on the MS-COCO dataset. Based on the paper [ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE](https://arxiv.org/pdf/1511.06361.pdf). The model can be trained with 2 approaches:

* Symmetric Distance(Cosine loss)
* Ordered Distance

Furthermore, the code is also capable of fine-tuning the model using the VIST dataset.

## Details

* Preprocessing(done by preproc.py): The first step is to extract out normalized 10-CROP VGG19 features from all the images. These features will be used for all further processing.

* Training(done by train.py): Loads the image-encoder and the sentence-encoder and trains using triplet loss alongwith Cosine Loss or orderEmbedding loss. No negative mining is being done in this code. The hyperparameters have been set according to the Research Paper.

* Testing(done by eval.py): Finds the mean retrieval rank against the test dataset.

Check utils.py for utility function definitions and model.py for encoder architecture.

## How to Run:

```bash
bar@foo$:~/ImageCaptionRetrieval/Symmetric embedding python3 preproc.py
bar@foo$:~/ImageCaptionRetrieval/Symmetric embedding python3 train.py
bar@foo$:~/ImageCaptionRetrieval/Symmetric embedding python3 eval.py
```
Similarly, the code can be run in Order embedding directory and FineTune directory.

## Performance

A mean rank of 80 out of 5000 images for a given caption can be easily achieved by training this model.

## Datasets

* [COCO](https://cocodataset.org/)
* [VIST](http://visionandlanguage.net/VIST/)

## Built Using

* [Python3](https://www.python.org)
* [PyTorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)

## Author

[Vaibhav Garg](https://github.com/VAIBHAV-2303)
