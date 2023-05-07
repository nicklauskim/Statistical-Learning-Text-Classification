# Machine Learning Methods for Text Classification in Python

Since the beginning of digital data, the easiest way to classify data is based on numeric keys. Nevertheless, with over two decades of digitized data, classification is complicated, especially when it comes to natural language data. Hence, the need for a classifier that identifies data based on text has become more evident in recent years. Text classification is a popular application of machine learning techniques that helps classify natural language data into different categories of interest to organize and clean databases. The purpose of this study is to test different text classification methods, evaluate their accuracy, and compare them to one another. Moreover, we aim to understand the simple classification approaches taught in class and how they might be applied specifically in the use case of text data. We will implement the Naive Bayes classifier (NB), Support Vector Machine (SVM), Random Forest (RF) and K-nearest neighbors (KNN) algorithms to classify different scientific research papers into various disciplines based on their titles. These are widely touted as some of the most popular and effective algorithms used in text classification in practice. Nowadays, different researchers may even use more elaborated and sophisticated methods, most notably large-scale neural networks and similar structures, to classify documents, emails, file and/or data into their proper categories. Nonetheless, such advanced and computationally expensive methods are outside the scope of this study.

Text classification techniques require thorough preprocessing of the data given that the natural language constituting the source data — English in this case — can contain many words with the same meaning that share the same word core (or root). Another obstacle to overcome is transforming the raw text into feature vectors (i.e. numeric variables) via text vectorization. This paper covers two of the most commonly used vectorization methods; (1) Count and (2) Term Frequency-Inverse Document- Frequency (TF-IDF). Count vectorizer will create a vector of frequencies that show how often the word appears in the text. The second, TF-IDF, will use a more elaborate measure, the multiplication of the word frequencies with a weighting factor. More details about the preprocessing, text vectorization, and classification models will be explained in later sections.

The original idea for this project was to replicate the findings from "Survey on supervised machine learning techniques for automatic text classification" by Kadhim (2019), but retrieving the author’s original data proved to be quite complicated. Therefore, we created our own database and limited the study to comparing the efficiency of the methods implemented by the aforementioned author in their experiment versus our own. For instance, according to the previously mentioned paper, NB should be able to outperform the SVM method if the dataset is significantly small; otherwise, methods like SVM and KNN should have a higher accuracy. We investigate this relationship as well as others across different methods by cross-examining the author’s findings with our own.

The approach implemented in this study has its limitations. Based on "Research paper classification systems based on TF-IDF and LDA schemes" by Kim et al., (2019) creating a classification model based on only the titles of the articles might not be enough in some circumstances. A more robust approach would be to include the abstracts, keywords, and sometimes even the entire body of the paper. Nonetheless, due to computational limitations we could not acquire all the papers in their totality, hence we only used the titles for feasibility.
