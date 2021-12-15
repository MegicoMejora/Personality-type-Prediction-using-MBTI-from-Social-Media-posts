# Personality-type-Prediction-using-MBTI-from-Social-Media-posts
The aim of this project is to build a classifier that can predict the personality of the author by analysing the texts extracted from his/her social media posts. 
The model used for personality analysis is Myers-Briggs Type Indicator (MBTI) method. MBTI model is the most popular and is widely used owing to the fact that hundreds of studies
over the past 40 years have proven this instrument to be both valid and reliable. Furthermore, this project proposes a deep learning architecture combined with Bidirectional Encoder Representations from Transformers (BERT), a pre-trained language representation model used to extract contextualized word embeddings from textual data for author’s personality detection. This project uses multiple social media data sources namely PersonalityCafe and Reddit and produces a predictive model for each trait using Natural Language Processing (NLP) and bidirectional context feature methods. The experiment is related to research with four binary classifiers that classifies MBTI type classification.

**The implementation of this project is carried out in by the following steps:**

1.	Data collection.
2.	Converting the dataset into four binary datasets.
3.	Text mining to transform the unstructured input text into a structured data by parsing and cleaning the data by removing unimportant elements.
4.  Data Augmentation
5.	Building the model.
6.	Evaluating the model.

## **1. Data Collection**

This project uses two datasets: 
1. The first dataset used is publicly available MBTI dataset from Kaggle. This data has been collected from the PersonalityCafe forum. This dataset contains 8675 rows of data. In this dataset each row is a person’s MBTI personality type, a combination of four labels and a section of fifty posts obtained from the individual’s social media. Each post has been separated by three pipe characters.

2. The second dataset used is MBTI9K – Corpus of the Reddit comments and posts labeled with MBTI personality types. It has 9149 rows and 7 columns from which only around 5000 rows and 2 columns namely ‘comments’ and ‘type’ were used for this research. This dataset was acquired on request from Matej Gjurković and Jan Šnajder. MBTI9K dataset is a subset of dataset containing the comments of authors who contributed with more than 1000 words and comments of each user annotated with the MBTI type of the author from the Reddit social media platform.
Reddit dataset can be requested from **'matej.gjurkovic@fer.hr'** and can refer the paper from **'https://aclanthology.org/W18-1112.pdf'**

## **2.	Converting the dataset into four binary datasets**

This project’s research is predominantly focused to build four binary classifiers that classifies each of the individual’s post as Introvert(I)/Extrovert(E), Sensing(S)/Intuition(N), Thinking(T)/Feeling(F) and Judging(J)/Perceiving(P). From dataset, the column ‘type’ is the combination of MBTI traits labelled for each individual’s posts. In order to obtain four binary classifiers, it is imperative to split each dataset into four binary datasets.  

Using Python string split() method, column ‘type’ is split into four columns. Then created a first separate binary dataset dichotomy of Introvert(I) Vs Extrovert(E) by concatenating first column of ‘type’ with the column ‘posts’. The second binary dataset dichotomy of Sensing(S) Vs Intuition(I) was created by concatenating second column of ‘type’ with the column ‘posts’. The third binary dataset dichotomy of Thinking(T) Vs Feeling(F) was created by concatenating third column of ‘type’ with the column ‘posts’. The final binary dataset dichotomy of Judging(J) Vs Perceiving(P) was created by concatenating the fourth column of ‘type’ with the posts. The sample of partitioned binary datasets of (I/E), (S/N), (T/F) and (J/P) 

## **3. Data preprocessing**

As the dataset was collected from social media, it contains lot of insignificant words which may not be a meaningful feature to identify a person’s personality and hence needs to be cleaned. The flow of the preprocess for the datasets are illustrated in Fig. 8.
•	Removing URL’s: The comments contain number of websites. Since the model is to be generalized to English language, the links to the websites are removed. 

•	Removing punctuation and special characters: Using Regular expression, punctuation and special characters are removed which appear rarely in the sentences to improve the performance of the model. 

•	Removing numbers: Numbers are removed as it doesn’t provide any useful information for the text. 

•	Expanding a contraction in the sentence such as the use of ‘you’re’ to make it ‘you are’.

•	Filtering non-words: Words without meaning or no occurrence in any text corpus or dictionary are removed using NLTK’s word punkt tokenizer. 

•	Converting each sentence into lower case. 

•	Removing stop-words: removing the stop-words (e.g., commonly used filter words like ‘a’, ‘the’, ‘is’, etc.,) from the text using Python’s NLTK text processing library and giving more focus to the important words. 

•	Removing Personality type words: Since the  datasets used in this project come from the websites intended for explicit discussion of MBTI type personality, it is important to remove the personality types from the posts to get valid model accuracy estimation for unseen data.

•	Removing words with length less than 3. 

•	Lemmatisation: Lemmatisation is done by using WordNetLemmatizer that reduces words to its root word. Lemmatisation is done because when reduced to root word it produces a dictionary meaning word (e.g., words ‘played’, ‘plays’, ‘playing’ all become ‘play’). This inflected forms, can be analyzed as a same word carrying one shared meaning and so as to  provide more accuracy.

•	Encoding:  Encoding converts the binary class labels associated with each data into 0/1.

The two datasets were preprocessed using the above explained techniques.

## **4. Data Augmentation**

When classes are imbalanced, classifiers predict the majority classes and the minority classes may be failed to predict. To avoid this, data augmentation is done for the training data using ‘nlpaug’ library for the MBTI Reddit dataset. The ’nlpaug’ library provides Contextual Word Embeddings for BERT language model. It has two actions namely Insertion and Substitution. Insertion is predicted by BERT language model rather than picking one word randomly and Substitution use surrounding words as a feature to predict the target word. 
For this study, the Substitution action was chosen and generated four augmented binary datasets. Data augmentation doubled the minority class present in the training data.

## **5. Model Building**

Model was built and trained, validated and tested. Finally, the models were evaluated with the results using multiple metrics such as accuracy, precision, recall, F1 score and AUC score, as the metrics suitable for imbalanced dataset. Also, while comparing the results it’s important to focus on F1-score as it measures the balance between the precision and recall. 
