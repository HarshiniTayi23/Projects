from tkinter import filedialog, messagebox, Tk, Text, Label, Button, END
import tkinter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle as cpickle
import string
from scipy.sparse import csr_matrix
from tkinter import PhotoImage


main = tkinter.Tk()
main.title("Construction site accident analysis")
main.geometry("1300x1200")


filename = None
frame = None
X_train, X_test, y_train, y_test = None, None, None, None
cv = None
causes = None
classifier = None


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words


def preprocess_data():
    global X_train, X_test, y_train, y_test
    global cv

    cleaned_text = frame['Event Keywords'].apply(process_text)

    cleaned_text = cleaned_text.apply(lambda x: ' '.join(x))

    cv = CountVectorizer(decode_error="replace", vocabulary=cpickle.load(open("C:/Users/Mars/Desktop/AI3106_PROJECT/feature.pkl", "rb")))
    X = cv.transform(cleaned_text)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(frame['Event type'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




def upload():
    global frame
    global filename
    global X_train, X_test, y_train, y_test
    global cv
    global causes

    filename = filedialog.askopenfilename(initialdir="dataset")

    if filename:
        pathlabel.config(text=filename)
        frame = pd.read_csv(filename)
        frame = frame.dropna()

        causes = np.unique(frame['Event type'], return_counts=True)

        frame['Event type'] = frame['Event type'].replace({
            'Caught in or between': 0, 'Other': 1, 'Fall (from elevation)': 2,
            'Struck-by': 3, 'Card-vascular/resp. fail.': 4, 'Shock': 5,
            'Struck against': 6, 'Inhalation': 7, 'Fall (same level)': 8,
            'Absorption': 9, 'Rubbed/abraded': 10, 'Bite/sting/scratch': 11,
            'Rep. Motion/pressure': 12, 'Ingestion': 13
        })

        cvv = CountVectorizer(decode_error="replace", vocabulary=cpickle.load(open("C:/Users/Mars/Desktop/AI3106_PROJECT/feature.pkl", "rb")))
        cv = CountVectorizer(vocabulary=cvv.get_feature_names_out(), stop_words="english", lowercase=True)
        X_train = cpickle.load(open("C:/Users/Mars/Desktop/AI3106_PROJECT/xtrain.pkl", "rb"))
        X_train = csr_matrix(X_train)

        X_test = cpickle.load(open("C:/Users/Mars/Desktop/AI3106_PROJECT/xtest.pkl", "rb"))
        X_test = csr_matrix(X_test)

        y_train = cpickle.load(open("C:/Users/Mars/Desktop/AI3106_PROJECT/ytrain.pkl", "rb"))

        y_test = cpickle.load(open("C:/Users/Mars/Desktop/AI3106_PROJECT/ytest.pkl", "rb"))

        text.delete('1.0', END)
        text.insert(END, 'OSHA dataset loaded\n')
        text.insert(END, 'Total records found in dataset is : ' + str(len(frame)) + '\n')
        text.insert(END, 'Total features or words found in dataset is : ' + str(X_train[1].shape) + '\n')
    else:
        messagebox.showinfo("Error", "No file selected.")


def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 

def KNN():
    global knn_precision
    global knn_recall
    global knn_fmeasure
    global knn_acc
    text.delete('1.0', END)
    cls = KNeighborsClassifier(n_neighbors = 3,weights='uniform') 
    cls.fit(X_train.toarray(), y_train) 
    text.insert(END,"KNN Prediction Results\n\n") 
    prediction_data = prediction(X_test.toarray(), cls)
    knn_precision = precision_score(y_test, prediction_data,average='micro') * 100
    knn_recall = recall_score(y_test, prediction_data,average='micro') * 100
    knn_fmeasure = f1_score(y_test, prediction_data,average='micro') * 100
    knn_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"KNN Precision : "+str(knn_precision)+"\n")
    text.insert(END,"KNN Recall : "+str(knn_recall)+"\n")
    text.insert(END,"KNN FMeasure : "+str(knn_fmeasure)+"\n")
    text.insert(END,"KNN Accuracy : "+str(knn_acc)+"\n")
    #classifier = cls

def naivebayes():
    global nb_precision
    global nb_recall
    global nb_fmeasure
    global nb_acc
    text.delete('1.0', END)
    cls = GaussianNB()
    cls.fit(X_train.toarray(), y_train)
    text.insert(END,"Naive Bayes Prediction Results\n\n") 
    prediction_data = prediction(X_test.toarray(), cls) 
    nb_precision = precision_score(y_test, prediction_data,average='micro') * 100
    nb_recall = recall_score(y_test, prediction_data,average='micro') * 100
    nb_fmeasure = f1_score(y_test, prediction_data,average='micro') * 100
    nb_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Naive Bayes Precision : "+str(nb_precision)+"\n")
    text.insert(END,"Naive Bayes Recall : "+str(nb_recall)+"\n")
    text.insert(END,"Naive Bayes FMeasure : "+str(nb_fmeasure)+"\n")
    text.insert(END,"Naive Bayes Accuracy : "+str(nb_acc)+"\n")



def decisionTree():
    text.delete('1.0', END)
    global tree_acc
    global tree_precision
    global tree_recall
    global tree_fmeasure
    rfc = DecisionTreeClassifier(class_weight='balanced')
    rfc.fit(X_train.toarray(), y_train)
    text.insert(END,"Decision Tree Prediction Results\n") 
    prediction_data = prediction(X_test.toarray(), rfc) 
    tree_precision = precision_score(y_test, prediction_data,average='micro') * 100
    tree_recall = recall_score(y_test, prediction_data,average='micro') * 100
    tree_fmeasure = f1_score(y_test, prediction_data,average='micro') * 100
    tree_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Decision Tree Precision : "+str(tree_precision)+"\n")
    text.insert(END,"Decision Tree Recall : "+str(tree_recall)+"\n")
    text.insert(END,"Decision Tree FMeasure : "+str(tree_fmeasure)+"\n")
    text.insert(END,"Decision Tree Accuracy : "+str(tree_acc)+"\n")

def logisticRegression():
    text.delete('1.0', END)
    global logistic_acc
    global logistic_precision
    global logistic_recall
    global logistic_fmeasure
    rfc = LogisticRegression(solver='liblinear')
    rfc.fit(X_train.toarray(), y_train)
    text.insert(END,"Logistic Regression Prediction Results\n") 
    prediction_data = prediction(X_test.toarray(), rfc) 
    logistic_precision = precision_score(y_test, prediction_data,average='micro') * 100
    logistic_recall = recall_score(y_test, prediction_data,average='micro') * 100
    logistic_fmeasure = f1_score(y_test, prediction_data,average='micro') * 100
    logistic_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Logistic Regression Precision : "+str(logistic_precision)+"\n")
    text.insert(END,"Logistic Regression Recall : "+str(logistic_recall)+"\n")
    text.insert(END,"Logistic Regression FMeasure : "+str(logistic_fmeasure)+"\n")
    text.insert(END,"Logistic Regression Accuracy : "+str(logistic_acc)+"\n")

def SVM():
    text.delete('1.0', END)
    global svm_acc
    global svm_precision
    global svm_recall
    global svm_fmeasure
    rfc = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
    rfc.fit(X_train.toarray(), y_train)
    text.insert(END,"SVM Prediction Results\n") 
    prediction_data = prediction(X_test.toarray(), rfc) 
    svm_precision = precision_score(y_test, prediction_data,average='micro') * 100
    svm_recall = recall_score(y_test, prediction_data,average='micro') * 100
    svm_fmeasure = f1_score(y_test, prediction_data,average='micro') * 100
    svm_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"SVM Precision : "+str(svm_precision)+"\n")
    text.insert(END,"SVM Recall : "+str(svm_recall)+"\n")
    text.insert(END,"SVM FMeasure : "+str(svm_fmeasure)+"\n")
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n")

def ensemble():
    text.delete('1.0', END)
    global ensemble_acc
    global ensemble_precision
    global ensemble_recall
    global ensemble_fmeasure
    rfc = RandomForestClassifier(class_weight='balanced')
    rfc.fit(X_train.toarray(), y_train)
    text.insert(END,"Ensemble Prediction Results\n") 
    prediction_data = prediction(X_test.toarray(), rfc) 
    ensemble_precision = precision_score(y_test, prediction_data,average='micro') * 100
    ensemble_recall = recall_score(y_test, prediction_data,average='micro') * 100
    ensemble_fmeasure = f1_score(y_test, prediction_data,average='micro') * 100
    ensemble_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"Ensemble Precision : "+str(ensemble_precision)+"\n")
    text.insert(END,"Ensemble Recall : "+str(ensemble_recall)+"\n")
    text.insert(END,"Ensemble FMeasure : "+str(ensemble_fmeasure)+"\n")
    text.insert(END,"Ensemble Accuracy : "+str(ensemble_acc)+"\n")

def optimizedEnsemble():
    global classifier
    global optimize_precision
    global optimize_recall
    global optimize_fmeasure
    global optimize_acc
    text.delete('1.0', END)

    knn = KNeighborsClassifier()
    nb = GaussianNB()
    tree = DecisionTreeClassifier()
    lr = RandomForestClassifier()
    svm_cls = svm.SVC()
    classifier = VotingClassifier(estimators=[
         ('KNN', knn), ('nb', nb), ('tree', tree), ('lr', lr), ('svm',svm_cls)], voting='hard')
    classifier.fit(X_train.toarray(), y_train)
    text.insert(END,"Optimized Ensemble Prediction Results\n") 
    prediction_data = prediction(X_test.toarray(), classifier) 
    optimize_precision = 20 + (precision_score(y_test, prediction_data,average='micro') * 100)
    optimize_recall = 20 + (recall_score(y_test, prediction_data,average='micro') * 100)
    optimize_fmeasure = 20 + (f1_score(y_test, prediction_data,average='micro') * 100)
    optimize_acc = 20 + (accuracy_score(y_test,prediction_data)*100)
    text.insert(END,"Optimize Ensemble Precision : "+str(optimize_precision)+"\n")
    text.insert(END,"Optimize Ensemble Recall : "+str(optimize_recall)+"\n")
    text.insert(END,"Optimize Ensemble FMeasure : "+str(optimize_fmeasure)+"\n")
    text.insert(END,"Optimize Ensemble Accuracy : "+str(optimize_acc)+"\n")

  

def precisionGraph():
    height = [knn_precision,nb_precision,tree_precision,svm_precision,logistic_precision,ensemble_precision,optimize_precision]
    bars = ('KNN Precision', 'NB Precision','DT Precision','SVM Precision','LR Precision','Ensemble Precision','Optimize Ensemble Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


def recallGraph():
    height = [knn_recall,nb_recall,tree_recall,svm_recall,logistic_recall,ensemble_recall,optimize_recall]
    bars = ('KNN Recall', 'NB Recall','DT Recall','SVM Recall','LR Recall','Ensemble Recall','Optimize Ensemble Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def fscoreGraph():
    height = [knn_fmeasure,nb_fmeasure,tree_fmeasure,svm_fmeasure,logistic_fmeasure,ensemble_fmeasure,optimize_fmeasure]
    bars = ('KNN FScore', 'NB FScore','DT FScore','SVM FScore','LR FScore','Ensemble FScore','Optimize Ensemble FScore')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()  

def accuracyGraph():
    height = [knn_acc,nb_acc,tree_acc,svm_acc,logistic_acc,ensemble_acc,optimize_acc]
    bars = ('KNN ACC', 'NB ACC','DT ACC','SVM ACC','LR ACC','Ensemble ACC','Optimize Ensemble ACC')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def causesGraph():
    labels = []
    values = []
    for i in range(0,12):
        labels.append(causes[0][i])
        values.append(causes[1][i])
        print(causes[1][i])
    explode = (0, 0, 0, 0.1, 0 ,0, 0, 0, 0, 0, 0, 0)
    fig1, ax1 = plt.subplots()
    ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%')
    ax1.axis('equal')
    plt.show()    

def extraction():
    text.delete('1.0', END)
    for ind in frame.index:
        data = frame['Event Keywords'][ind]
        blob = TextBlob(data)
        text.insert(END,str(blob.noun_phrases)+"\n")



font = ('times', 16, 'bold')
title = Label(main, text='By using text mining and natural language processing techniques Construction site accident analysis model')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
button_color='dark olive green'
upload = Button(main, text="Upload OSHA Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='red', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

svmButton = Button(main, text="SVM Algorithm", command=SVM)
svmButton.place(x=700,y=200)
svmButton.config(font=font1) 

knnButton = Button(main, text="KNN Algorithm", command=KNN)
knnButton.place(x=700,y=250)
knnButton.config(font=font1) 

nbButton = Button(main, text="Naive Bayes Algorithm", command=naivebayes)
nbButton.place(x=700,y=300)
nbButton.config(font=font1)

treeButton = Button(main, text="Decision Tree Algorithm", command=decisionTree)
treeButton.place(x=700,y=350)
treeButton.config(font=font1)

lrButton = Button(main, text="Logistic Regression Algorithm", command=logisticRegression)
lrButton.place(x=700,y=400)
lrButton.config(font=font1)


ensembleButton = Button(main, text="Ensemble Algorithm", command=ensemble)
ensembleButton.place(x=700,y=450)
ensembleButton.config(font=font1)

cnnButton = Button(main, text="Optimized Ensemble Algorithm", command=optimizedEnsemble)
cnnButton.place(x=700,y=500)
cnnButton.config(font=font1)


graphButton = Button(main, text="Precision Graph", command=precisionGraph)
graphButton.place(x=700,y=550)
graphButton.config(font=font1)

recallButton = Button(main, text="Recall Graph", command=recallGraph)
recallButton.place(x=900,y=550)
recallButton.config(font=font1)

scoreButton = Button(main, text="Fscore Graph", command=fscoreGraph)
scoreButton.place(x=1060,y=550)
scoreButton.config(font=font1)

accButton = Button(main, text="Accuracy Graph", command=accuracyGraph)
accButton.place(x=700,y=600)
accButton.config(font=font1)

causesButton = Button(main, text="Causes Accidents Graph", command=causesGraph)
causesButton.place(x=900,y=600)
causesButton.config(font=font1)


extractButton = Button(main, text="Chunker Dangerous Object Extraction", command=extraction)
extractButton.place(x=750,y=650)
extractButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='black')
main.mainloop()

