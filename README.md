# Multi-Class Sentimental Analysis with Deep Learning #
## 5-class Sentimental Analysis: LSTM Architecture Comparison## 

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/sentimental_bot.gif">
</p>

## Purpose ##

To use **different LSTM architectures having different number of layers and regularization tweaks to do multi-class sentimental analysis.** Accuracy and loss are being analyzed to compare these architectures. The implementation is done in Keras.

## Steps at a Glance: ##

1. Take Amazon Review dataset as input, as it contains 5 level rating along with review text.

2. Generate a vocabulary of all words

3. Make a word-frequency table having frequency corresponding to each word

4. Generate the index of each word based on sorted frequency (only top ‘n’ words are considered)

5. Encode the reviews as a set of indices of top ‘n’ frequent words. Remaining words are ignored.

6. Run the LSTM Model on Single Layer & Double-Layer LSTM, each layer having 100s of LSTMs stacked in parallel.

7. Tune for higher Accuracy by changing # of neurons in each layer to compare performance of different architectures.

8. Draw error plots, of both train &  test loss, for each architecture to find whether the model is overfitting or not.

9. Apply regularization such as Dropout, L1, L2, L1L2 or a combination of these to reduce overfitting.

10. Conclusion based on the accuracy and plots obtained with test data.


## Data Source ## 

Amazon Fine Food Review Dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews

The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.<br>
Number of reviews : 568,454<br>
Number of users : 256,059<br>
Number of products : 74,258<br>
Timespan: Oct 1999 : Oct 2012<br>
Number of Attributes/Columns in data: 10 <br>

    

## Model 1: Single Layer LSTM Architecture: 100 (1) LSTM stack ##

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/M1_Accuracy.png">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/M1_Loss.png">
</p>


## Model 2: Multiple Layer LSTM Architecture: 200 (1) -150 (2) LSTM stack ##

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/M2_Accuracy.png">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/M2_Loss.png">
</p>

### Model 3: Multi-Layer Neuron-Dense LSTM Architecture: 512 (1) -256 (2) LSTM stack ###

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/M3_Accuracy.png">
</p>

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/M2_Loss.png">
</p>



## Summary Statistics ##

<p align="center">
    <img src="https://github.com/AdroitAnandAI/Multi-Class-Sentimental-Analysis-Deep-Learning/blob/main/summary.png">
</p>


## Conclusions ##

1. Three architectures with single layer and double layer LSTMs are used to train frequency-encoded Amazon Review dataset.

2. **Double-Layer LSTM Architecture obtained highest accuracy** on validation dataset.

3. **A single layer stack of 100 LSTMs (M1) fetched a commendable validation accuracy of 89.99%.**

4. The validation accuracy of multi layer neuron dense LSTM stack (M3) fell to 89.97%, though it showed a hike in training accuracy, 93.7%. The increase in training accuracy and reduction in test accuracy **points to overfitting on the train data.**

5. The slight improvement in accuracy of Model 2 may not be worth the extra time spent on training such a stack-dense model. However, for higher accuracy **Model 2 can be used.**
