# Yelp Review Word2vec

這是根據 Yelp 資料集中的 Review 內的內容所作的 word2vec ～



## 資料集
資料集內容可以參考[Yelp Dataset](https://www.yelp.com/dataset)，有針對 Dataset 的內容說明

這裡所使用的是其中一個 review.json 內 text 欄位的內容來作 word2vec

## 程式
主要分成以下幾個部分：

第一個部分 : 清理文字內容，可以參考 yelp_review_preprocess.py 
             （不使用stopword，因此將該段註解掉)

第二個部分 : 建立 word2vec 模型，為了方便執行，因此分成兩個

         * CBOW(Continuous bag of words) : yelp_review_w2v_cbow.py

         * Skip-Gram : yelp_review_w2v_skgram.py

第三個部分 : 進行檢視訓練後的結果，可以參考 yelp_review.ipynb

## 模型
1. CBOW(Continuous bag of words) : [yelp_cbow](https://drive.google.com/open?id=1gBm_5wH9j4wAiwGCwRwwcvtzrzDE1Ngi)

2. Skip-Gram : [yelp_skipgram](https://drive.google.com/open?id=1gBm_5wH9j4wAiwGCwRwwcvtzrzDE1Ngi)


清理資料時有使用 stop words 後所建立的模型，

1. CBOW(Continuous bag of words) : [yelp_cbow_stopwords](https://drive.google.com/open?id=1UxZq5YQRETMXvIh_LTn0S0dmeqtnX_tL)

2. Skip-Gram : [yelp_skipgram_stopwords](https://drive.google.com/open?id=1UxZq5YQRETMXvIh_LTn0S0dmeqtnX_tL)



## 參考資料
1. [Yelp Dataset](https://www.yelp.com/dataset)
2. [https://www.kaggle.com/vksbhandary/exploring-yelp-reviews-dataset](https://www.kaggle.com/vksbhandary/exploring-yelp-reviews-dataset)
3. [https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f](https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f)
4. [https://www.kaggle.com/liananapalkova/simply-about-word2vec/notebook](https://www.kaggle.com/liananapalkova/simply-about-word2vec/notebook)
5. [https://www.kaggle.com/itratrahman/nlp-tutorial-using-python](https://www.kaggle.com/itratrahman/nlp-tutorial-using-python)

