# Import libraries
from math import log
# Data manipulation
import numpy as np
import pandas as pd
import string
import re
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Natural langauge processing
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


# Set plot settings for later:
plt.style.use('fivethirtyeight')
sns.set_palette('muted')

# Set font sizes for plots:
plt.rc('font', size = 12)
plt.rc('axes', labelsize = 8)
plt.rc('legend', fontsize = 18)
plt.rc('axes', titlesize = 14)
plt.rc('figure', titlesize = 18)


# Import data:
#os.chdir("C:/Users/maria/Desktop/UCLA STATS 201B/ST 201B project")
stats = pd.read_csv('~/Documents/Winter 2022/STATS 201B/Final Project/data/resultstats.csv')
educ = pd.read_csv('~/Documents/Winter 2022/STATS 201B/Final Project/data/resulteducation.csv')
civeng = pd.read_csv('~/Documents/Winter 2022/STATS 201B/Final Project/data/resultcivileng.csv')
bio = pd.read_csv('~/Documents/Winter 2022/STATS 201B/Final Project/data/resultbiology.csv')


#Helper functions to be used later:
def preprocess(text):
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  # List of common stopwords in the english language
    stop_words.update(['im', 'let', 'us', '', 'исследований', 'области', 'образования', 'αstabl', 'ℓnorm', 'ℓ-penal',
                       'ℓregular', 'ℓ', 'μct', '以工程认证和审核评估为契机', '全面深化专业综合改革', '略论应用型本科院校的定位', 'в']) # Add some words we want to eliminate that might not be in the list
    stemmer = nltk.PorterStemmer()
    text_lc = ''.join([word.lower() for word in text if word not in string.punctuation])  # tokenize + make lowercase
    text_rc = re.sub(r'\d+', '', text_lc)  # remove punctuation, other non-letter characters
    tokens = re.split('\W+', text_rc)  # tokenization
    text_cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]  # remove stopwords and stemming
    return text_cleaned

def find_likelihood(word, category):
    """
    Takes as inputs:
    word = which word we want the conditional probability for;
    df = the dataframe of counts we will use (e.g. vect_stats)
    """
    # Follow procedure as before, getting the (subset of the) vectorized df for only 1 category
    category_df = vect_train.loc[np.array(Y_train == category), :]
    n = category_df.shape[0]
    return (category_df[word].sum() + 1) / (n + 1)

def naivebayes(title, categories):
    """
    Takes as inputs a title and a category we want to find the posterior for (both string inputs)
    - Make sure to use proper name of category
    Finds a single probability, denoting the posterior probability, i.e. P(title|category)
    Gives as output a table/dataframe storing the posteriors for every category so we can compare values easily
    """
    # Initialize empty df to fill and return later
    df_nb = pd.DataFrame(np.zeros((1, len(categories))), columns=categories)
    # Preprocess title to get list of words to check
    words = preprocess(title)
    # We want to find the posterior for every category - loop through list of category names
    for cat in categories:
        # Find (product of) likelihoods of the words found in the title (or sum of log probabs?)
        likelihood_product = 0
        for word in words:
            if word in freq_final.columns.values:
                likelihood_product += log(freq_final.loc[cat, word])
        # Multiply by prior (found earlier) to get posterior
        df_nb[cat] = log(priors[cat]) + likelihood_product
    return df_nb

def nb_classifier(X, Y, categories):
    """
    Decides which category to assign to a given paper title
    Then, checks against true class label...
    """
    n = X.shape[0]
    n_correct = 0
    cols = ['Failed', 'Real Category', 'Probs: Sta', 'Probs: Ci Eng', 'Probs: Edu', 'Probs: Bio']
    NB = []
    for i, paper in X.iterrows():
        pos = naivebayes(paper['Title'], categories) #Compute posteriors
        choice = pos.idxmax(axis=1)
        row = pd.Series(index=cols, dtype=object)
        if choice.values == Y.loc[i]:
            n_correct += 1
        else:
            row['Failed'] = paper['Title']
            row['Real Category'] = Y.loc[i]
            row['Probs: Sta'] = str(round(pos['Statistics'][0],3))
            row['Probs: Ci Eng'] = str(round(pos['Civil Engineering'][0],3))
            row['Probs: Edu'] = str(round(pos['Education'][0],3))
            row['Probs: Bio'] = str(round(pos['Biology'][0],3))
            NB.append(row)
    ResultsNB = pd.DataFrame.from_dict(NB)
    predic = n_correct / n
    return ResultsNB, predic


# Create column with class labels for each df (e.g. 'Stats'):
stats['Category'] = stats['Journal'].apply(lambda x: 'Statistics')
civeng['Category'] = civeng['Journal'].apply(lambda x: 'Civil Engineering')
educ['Category'] = educ['Journal'].apply(lambda x: 'Education')
bio['Category'] = bio['Journal'].apply(lambda x: 'Biology')
df = pd.concat([stats, educ, civeng, bio], ignore_index = True)[['Name', 'Category']] # Combine the individual dfs into one
df = df.rename(columns = {'Name': 'Title'}) # Rename column(s)

# Remove foreign titles
foreign_index = df['Title'].apply(lambda x: detect(x) == 'en')
df = df.loc[foreign_index, :]

# Separate df into multiple dfs by subject - mainly convenient for plotting
df_stats = df.loc[np.array(df['Category'] == 'Statistics'), :]
df_civeng = df.loc[np.array(df['Category'] == 'Civil Engineering'), :]
df_educ = df.loc[np.array(df['Category'] == 'Education'), :]
df_bio = df.loc[np.array(df['Category'] == 'Biology'), :]


# Get all words appearing in each category
all_stats_words = sorted(preprocess(df_stats['Title']))
all_civeng_words = sorted(preprocess(df_civeng['Title']))
all_educ_words = sorted(preprocess(df_educ['Title']))
all_bio_words = sorted(preprocess(df_bio['Title']))


# Get the 25 highest ranking words for plotting
words_dict = {k: v for k, v in sorted(dict(FreqDist(all_stats_words)).items(), key=lambda x: x[1], reverse=True)}
stats_keys = list(words_dict)[:30]
stats_values = [words_dict[key] for key in list(words_dict)[:30]]
words_dict = {k: v for k, v in sorted(dict(FreqDist(all_civeng_words)).items(), key=lambda x: x[1], reverse=True)}
civeng_keys = list(words_dict)[:30]
civeng_values = [words_dict[key] for key in list(words_dict)[:30]]
words_dict = {k: v for k, v in sorted(dict(FreqDist(all_educ_words)).items(), key=lambda x: x[1], reverse=True)}
educ_keys = list(words_dict)[:30]
educ_values = [words_dict[key] for key in list(words_dict)[:30]]
words_dict = {k: v for k, v in sorted(dict(FreqDist(all_bio_words)).items(), key=lambda x: x[1], reverse=True)}
bio_keys = list(words_dict)[:30]
bio_values = [words_dict[key] for key in list(words_dict)[:30]]


# Plot bar graph of most commonly appearing words for given category
figure = plt.figure()
sns.barplot(x=stats_keys, y=stats_values, palette='viridis')
plt.xticks(rotation=90)
plt.title("Most Common Statistics Words")

figure = plt.figure()
sns.barplot(x=civeng_keys, y=civeng_values, palette='viridis')
plt.xticks(rotation=90)
plt.title("Most Common Civil Engineering Words")

figure = plt.figure()
sns.barplot(x=educ_keys, y=educ_values, palette='viridis')
plt.xticks(rotation=90)
plt.title("Most Common Education Words")

figure = plt.figure()
sns.barplot(x=bio_keys, y=bio_values, palette='viridis')
plt.xticks(rotation=90)
plt.title("Most Common Biology Words")

# Create and generate a word cloud image:
fig, ax = plt.subplots(figsize = (20, 20))
text = " ".join(word for word in all_stats_words)
wordcloud = WordCloud(max_font_size = 40, max_words = 50, background_color = "white", width = 500, height = 300, collocations = False).generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

fig, ax = plt.subplots(figsize = (20, 20))
text = " ".join(word for word in all_civeng_words)
wordcloud = WordCloud(max_font_size = 40, max_words = 50, background_color = "white", width = 500, height = 300, collocations = False).generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()


# Split data into training set and test set
X = df[['Title']]
Y = df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


#Find the distribution of train and test data as well as how many classes in train and test data
PIE1 = sum(Y_train.str.count("Statistics")), sum(Y_train.str.count("Civil Engineering")), sum(Y_train.str.count("Education")), sum(Y_train.str.count("Biology"))
PIE2 = sum(Y_test.str.count("Statistics")), sum(Y_test.str.count("Civil Engineering")), sum(Y_test.str.count("Education")), sum(Y_test.str.count("Biology"))
PIE3 = len(Y_train), len(Y_test)


# Plot pie chart to compare how much data we have in train and test, as well as the distribution between classes
colors = ("orange", "cyan", "brown","grey")
labels = 'Stats', 'Civil Eng', 'Edu', 'Biol'
colors1 = ("orange", "cyan")
labels1 = 'Train Data', 'Test Data'
fig = plt.figure()
#ax1 = plt.subplot2grid((1,2),(0,0))
plt.pie(PIE1,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Classes in Train Data:'+str(sum(PIE1)))
fig = plt.figure()
#ax1 = plt.subplot2grid((1,2),(0,1))
plt.pie(PIE2,colors=colors, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Classes in Test Data: '+str(sum(PIE2)))
fig = plt.figure()
plt.pie(PIE3,colors=colors1, labels=labels1, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('Train vs Test Data: '+str(sum(PIE3)))


# Naive Bayes Classifier:
print(" *********** Results from Naive Bayes: *********** ")
for i in range(0,2):
    if i == 0: #i = 0: CountVectorizer
        print(" -------- Results from CountVectorizer: -------- ")
        Vectorizer = CountVectorizer(analyzer = preprocess)
        # Vectorizer.fit(df['Title'])
        Vectorizer.fit(X_train['Title'])
    else:  #i = 1: TfidfVectorizer
        print(" -------- Results from TF-IDFVectorizer: -------- ")
        Vectorizer = TfidfVectorizer(analyzer = preprocess)
        # Vectorizer.fit(df['Title'])
        Vectorizer.fit(X_train['Title'])

    # print("In the dataset overall, we have {} paper titles with {} (unique) words in total".format(text_vect.shape[0], text_vect.shape[1]))
    text_vect_train = Vectorizer.transform(X_train['Title'])
    text_vect_test = Vectorizer.transform(X_test['Title'])

    # Create dataframes (both train and test) of vectorized text
    vect_train = pd.DataFrame(text_vect_train.toarray(), columns = Vectorizer.get_feature_names())
    vect_test = pd.DataFrame(text_vect_test.toarray(), columns = Vectorizer.get_feature_names())

    # Split vect_df_train into 4 dfs, one for each category we have
    vect_stats = vect_train.loc[np.array(Y_train == 'Statistics'), :]
    vect_civeng = vect_train.loc[np.array(Y_train == 'Civil Engineering'), :]
    vect_educ = vect_train.loc[np.array(Y_train == 'Education'), :]
    vect_bio = vect_train.loc[np.array(Y_train == 'Biology'), :]

    # Find prior probabilities
    prior_sta = (Y_train == 'Statistics').sum() / Y_train.shape[0]
    prior_cie = (Y_train == 'Civil Engineering').sum() / Y_train.shape[0]
    prior_edu = (Y_train == 'Education').sum() / Y_train.shape[0]
    prior_bio = (Y_train == 'Biology').sum() / Y_train.shape[0]
    check = prior_sta + prior_cie + prior_edu + prior_bio # Check the priors
    # print("The sum of the prior probabilities most be 1, we got: {}.".format(check))
    priors = pd.DataFrame([prior_sta, prior_cie, prior_edu, prior_bio]).transpose() # Store prior probabilities in convenient df
    categories = ['Statistics', 'Civil Engineering', 'Education', 'Biology']
    priors.columns = categories

    # Find word frequencies in each category using training set - these will be the likelihoods when doing naive bayes
    # +1 to assume every word appears in every category...
    freq_stats = pd.DataFrame((vect_stats.sum(axis = 0)+1) / vect_stats.shape[0]).transpose()
    freq_civeng = pd.DataFrame((vect_civeng.sum(axis = 0)+1) / vect_civeng.shape[0]).transpose()
    freq_educ = pd.DataFrame((vect_educ.sum(axis = 0)+1) / vect_educ.shape[0]).transpose()
    freq_bio = pd.DataFrame((vect_bio.sum(axis = 0)+1) / vect_bio.shape[0]).transpose()

    freq_final = pd.concat([freq_stats, freq_civeng, freq_educ, freq_bio], ignore_index = True)
    freq_final.rename(index = {0:'Statistics', 1:'Civil Engineering', 2:'Education', 3:'Biology'}, inplace=True)

    # Naive Bayes from function Nick created:
    ResultsNB, predic = nb_classifier(X_test, Y_test, categories)
    if i == 0:
        ResultsNB_Count = ResultsNB
        ResultsNB_Count.to_csv('ResultsNB_Count.csv')
    else:
        ResultsNB_TFIDF = ResultsNB
        ResultsNB_TFIDF.to_csv('ResultsNB_TFIDF.csv')
    print("NB Accuracy Score: {}.".format(predic))


# SVM and Random Forest from sklearn
print(" *********** Results from SVM and Random Forest: *********** ")
for i in range(0,2):
    if i == 0: #i = 0: CountVectorizer
        print(" -------- Results from CountVectorizer: -------- ")
        Vectorizer = CountVectorizer(analyzer = preprocess)
        Vectorizer.fit(X_train['Title'])
    else:  #i = 1: TfidfVectorizer
        print(" -------- Results from TF-IDFVectorizer: -------- ")
        Vectorizer = TfidfVectorizer(analyzer = preprocess)
        Vectorizer.fit(X_train['Title'])

    text_vect_train = Vectorizer.transform(X_train['Title'])
    text_vect_test = Vectorizer.transform(X_test['Title'])

    text_vect_train = Normalizer().fit_transform(text_vect_train) #Normalization (mean = 0, st dev = 1)
    text_vect_test = Normalizer().fit_transform(text_vect_test)
    
    if i == 0:
        svd = TruncatedSVD(n_components=500)
        x = np.arange(500)
    else:
        svd = TruncatedSVD(n_components=500)
        x = np.arange(500)
    text_vect_train = svd.fit_transform(text_vect_train) #Singular Value decomposition, i.e. dimension reduction
    text_vect_test = svd.transform(text_vect_test)

    fig = plt.figure()
    plt.plot(x, svd.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('SV')
    plt.ylabel('Variance Explained')
    plt.show()

    # Create dataframes (both train and test) of vectorized text
    vect_train = pd.DataFrame(text_vect_train)
    vect_test = pd.DataFrame(text_vect_test)

    # SVM from sklearn:
    clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    scores = cross_val_score(clf,vect_train,Y_train, cv = 5) #Cross-validation
    print("Validation score: ", scores.mean())
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test) # Predict the labels on the test dataset
    print("SVM Accuracy Score: ", accuracy_score(predictions, Y_test)) # Use accuracy_score function to get the accuracy

    #Random Forest from sklearn:
    clf = RandomForestClassifier()
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test)
    scores = cross_val_score(clf, vect_train, Y_train, cv=5) #Cross-validation
    print("Validation score: ", scores.mean())
    print("Random Forest Accuracy Score: ", accuracy_score(predictions, Y_test))

    #k-nearest neighbors from sklearn:
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(vect_train, Y_train)
    predictions = clf.predict(vect_test)
    scores = cross_val_score(clf, vect_train, Y_train, cv=5)  # Cross-validation
    print("Validation score: ", scores.mean())
    print("kNN Accuracy Score: ", accuracy_score(predictions, Y_test))
    
    
    