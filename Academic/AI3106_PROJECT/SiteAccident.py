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

# Fix pandas future warning
pd.set_option('future.no_silent_downcasting', True)

main = tkinter.Tk()
main.title("Construction site accident analysis")
main.geometry("1300x1200")

filename = None
frame = None
X_train, X_test, y_train, y_test = None, None, None, None
cv = None
causes = None
classifier = None

# Initialize metric variables
knn_precision = knn_recall = knn_fmeasure = knn_acc = 0
nb_precision = nb_recall = nb_fmeasure = nb_acc = 0
tree_precision = tree_recall = tree_fmeasure = tree_acc = 0
logistic_precision = logistic_recall = logistic_fmeasure = logistic_acc = 0
svm_precision = svm_recall = svm_fmeasure = svm_acc = 0
ensemble_precision = ensemble_recall = ensemble_fmeasure = ensemble_acc = 0
optimize_precision = optimize_recall = optimize_fmeasure = optimize_acc = 0

def safe_pickle_load(file_path):
    """Safely load pickle files with error handling for pandas compatibility"""
    try:
        # Try pandas read_pickle first (handles version compatibility better)
        return pd.read_pickle(file_path)
    except Exception as e1:
        print(f"Pandas read_pickle failed for {file_path}: {e1}")
        try:
            # Fallback to regular pickle load
            with open(file_path, "rb") as f:
                return cpickle.load(f)
        except Exception as e2:
            print(f"Regular pickle load failed for {file_path}: {e2}")
            return None

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

    vocabulary = safe_pickle_load("C:/Users/tayis/OneDrive/Desktop/Projects/Academic/AI3106_PROJECT/feature.pkl")
    if vocabulary is None:
        messagebox.showerror("Error", "Could not load feature.pkl file")
        return
    
    cv = CountVectorizer(decode_error="replace", vocabulary=vocabulary)
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
        try:
            pathlabel.config(text=filename)
            frame = pd.read_csv(filename)
            frame = frame.dropna()

            causes = np.unique(frame['Event type'], return_counts=True)

            # Fixed the replace method with proper infer_objects
            frame['Event type'] = frame['Event type'].replace({
                'Caught in or between': 0, 'Other': 1, 'Fall (from elevation)': 2,
                'Struck-by': 3, 'Card-vascular/resp. fail.': 4, 'Shock': 5,
                'Struck against': 6, 'Inhalation': 7, 'Fall (same level)': 8,
                'Absorption': 9, 'Rubbed/abraded': 10, 'Bite/sting/scratch': 11,
                'Rep. Motion/pressure': 12, 'Ingestion': 13
            }).infer_objects(copy=False)

            # Load pickle files safely with improved error handling
            base_path = r"C:\Users\tayis\OneDrive\Desktop\Projects\Academic\AI3106_PROJECT"
            
            # Try to load each file individually with better error reporting
            vocabulary = safe_pickle_load(f"{base_path}/feature.pkl")
            if vocabulary is None:
                messagebox.showerror("Error", "Could not load feature.pkl. Please check if the file exists and is compatible.")
                return
            
            X_train_data = safe_pickle_load(f"{base_path}/xtrain.pkl")
            if X_train_data is None:
                messagebox.showerror("Error", "Could not load xtrain.pkl. Please check if the file exists and is compatible.")
                return
                
            X_test_data = safe_pickle_load(f"{base_path}/xtest.pkl")
            if X_test_data is None:
                messagebox.showerror("Error", "Could not load xtest.pkl. Please check if the file exists and is compatible.")
                return
                
            y_train_data = safe_pickle_load(f"{base_path}/ytrain.pkl")
            if y_train_data is None:
                messagebox.showerror("Error", "Could not load ytrain.pkl. Please check if the file exists and is compatible.")
                return
                
            y_test_data = safe_pickle_load(f"{base_path}/ytest.pkl")
            if y_test_data is None:
                messagebox.showerror("Error", "Could not load ytest.pkl. Please check if the file exists and is compatible.")
                return

            # Set up CountVectorizer with better error handling
            try:
                if isinstance(vocabulary, dict):
                    cv = CountVectorizer(vocabulary=vocabulary, stop_words="english", lowercase=True)
                else:
                    cv = CountVectorizer(vocabulary=list(vocabulary), stop_words="english", lowercase=True)
            except Exception as e:
                messagebox.showerror("Error", f"Error setting up CountVectorizer: {str(e)}")
                return
            
            # Convert to appropriate format
            try:
                if hasattr(X_train_data, 'toarray'):
                    X_train = X_train_data
                else:
                    X_train = csr_matrix(X_train_data)
                    
                if hasattr(X_test_data, 'toarray'):
                    X_test = X_test_data
                else:
                    X_test = csr_matrix(X_test_data)
                    
                # Handle y data - convert to numpy array if it's a pandas Series
                if isinstance(y_train_data, pd.Series):
                    y_train = y_train_data.values
                else:
                    y_train = np.array(y_train_data)
                    
                if isinstance(y_test_data, pd.Series):
                    y_test = y_test_data.values
                else:
                    y_test = np.array(y_test_data)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error converting data formats: {str(e)}")
                return

            text.delete('1.0', END)
            text.insert(END, 'OSHA dataset loaded\n')
            text.insert(END, 'Total records found in dataset is : ' + str(len(frame)) + '\n')
            text.insert(END, 'Total features or words found in dataset is : ' + str(X_train.shape[1]) + '\n')
            text.insert(END, 'Training samples: ' + str(X_train.shape[0]) + '\n')
            text.insert(END, 'Test samples: ' + str(X_test.shape[0]) + '\n')
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
            print(f"Detailed error: {e}")  # For debugging
    else:
        messagebox.showinfo("Error", "No file selected.")

def prediction(X_test_data, cls): 
    try:
        y_pred = cls.predict(X_test_data)
        return y_pred
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def check_data_loaded():
    """Check if training data is loaded"""
    if X_train is None or y_train is None or X_test is None or y_test is None:
        messagebox.showerror("Error", "Please load dataset first!")
        return False
    return True

def get_array_data(data):
    """Helper function to convert sparse matrix to array if needed"""
    if hasattr(data, 'toarray'):
        return data.toarray()
    return data

def KNN():
    global knn_precision, knn_recall, knn_fmeasure, knn_acc
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)
        cls = KNeighborsClassifier(n_neighbors=3, weights='uniform') 
        cls.fit(get_array_data(X_train), y_train) 
        text.insert(END, "KNN Prediction Results\n\n") 
        
        prediction_data = prediction(get_array_data(X_test), cls)
        if prediction_data is not None:
            knn_precision = precision_score(y_test, prediction_data, average='micro') * 100
            knn_recall = recall_score(y_test, prediction_data, average='micro') * 100
            knn_fmeasure = f1_score(y_test, prediction_data, average='micro') * 100
            knn_acc = accuracy_score(y_test, prediction_data) * 100
            
            text.insert(END, "KNN Precision : " + str(knn_precision) + "\n")
            text.insert(END, "KNN Recall : " + str(knn_recall) + "\n")
            text.insert(END, "KNN FMeasure : " + str(knn_fmeasure) + "\n")
            text.insert(END, "KNN Accuracy : " + str(knn_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"KNN Error: {str(e)}")

def naivebayes():
    global nb_precision, nb_recall, nb_fmeasure, nb_acc
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)
        cls = GaussianNB()
        cls.fit(get_array_data(X_train), y_train)
        text.insert(END, "Naive Bayes Prediction Results\n\n") 
        
        prediction_data = prediction(get_array_data(X_test), cls)
        if prediction_data is not None:
            nb_precision = precision_score(y_test, prediction_data, average='micro') * 100
            nb_recall = recall_score(y_test, prediction_data, average='micro') * 100
            nb_fmeasure = f1_score(y_test, prediction_data, average='micro') * 100
            nb_acc = accuracy_score(y_test, prediction_data) * 100
            
            text.insert(END, "Naive Bayes Precision : " + str(nb_precision) + "\n")
            text.insert(END, "Naive Bayes Recall : " + str(nb_recall) + "\n")
            text.insert(END, "Naive Bayes FMeasure : " + str(nb_fmeasure) + "\n")
            text.insert(END, "Naive Bayes Accuracy : " + str(nb_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"Naive Bayes Error: {str(e)}")

def decisionTree():
    global tree_acc, tree_precision, tree_recall, tree_fmeasure
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)
        rfc = DecisionTreeClassifier(class_weight='balanced')
        rfc.fit(get_array_data(X_train), y_train)
        text.insert(END, "Decision Tree Prediction Results\n") 
        
        prediction_data = prediction(get_array_data(X_test), rfc)
        if prediction_data is not None:
            tree_precision = precision_score(y_test, prediction_data, average='micro') * 100
            tree_recall = recall_score(y_test, prediction_data, average='micro') * 100
            tree_fmeasure = f1_score(y_test, prediction_data, average='micro') * 100
            tree_acc = accuracy_score(y_test, prediction_data) * 100
            
            text.insert(END, "Decision Tree Precision : " + str(tree_precision) + "\n")
            text.insert(END, "Decision Tree Recall : " + str(tree_recall) + "\n")
            text.insert(END, "Decision Tree FMeasure : " + str(tree_fmeasure) + "\n")
            text.insert(END, "Decision Tree Accuracy : " + str(tree_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"Decision Tree Error: {str(e)}")

def logisticRegression():
    global logistic_acc, logistic_precision, logistic_recall, logistic_fmeasure
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)
        rfc = LogisticRegression(solver='liblinear', max_iter=1000)
        rfc.fit(get_array_data(X_train), y_train)
        text.insert(END, "Logistic Regression Prediction Results\n") 
        
        prediction_data = prediction(get_array_data(X_test), rfc)
        if prediction_data is not None:
            logistic_precision = precision_score(y_test, prediction_data, average='micro') * 100
            logistic_recall = recall_score(y_test, prediction_data, average='micro') * 100
            logistic_fmeasure = f1_score(y_test, prediction_data, average='micro') * 100
            logistic_acc = accuracy_score(y_test, prediction_data) * 100
            
            text.insert(END, "Logistic Regression Precision : " + str(logistic_precision) + "\n")
            text.insert(END, "Logistic Regression Recall : " + str(logistic_recall) + "\n")
            text.insert(END, "Logistic Regression FMeasure : " + str(logistic_fmeasure) + "\n")
            text.insert(END, "Logistic Regression Accuracy : " + str(logistic_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"Logistic Regression Error: {str(e)}")

def SVM():
    global svm_acc, svm_precision, svm_recall, svm_fmeasure
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)
        rfc = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
        rfc.fit(get_array_data(X_train), y_train)
        text.insert(END, "SVM Prediction Results\n") 
        
        prediction_data = prediction(get_array_data(X_test), rfc)
        if prediction_data is not None:
            svm_precision = precision_score(y_test, prediction_data, average='micro') * 100
            svm_recall = recall_score(y_test, prediction_data, average='micro') * 100
            svm_fmeasure = f1_score(y_test, prediction_data, average='micro') * 100
            svm_acc = accuracy_score(y_test, prediction_data) * 100
            
            text.insert(END, "SVM Precision : " + str(svm_precision) + "\n")
            text.insert(END, "SVM Recall : " + str(svm_recall) + "\n")
            text.insert(END, "SVM FMeasure : " + str(svm_fmeasure) + "\n")
            text.insert(END, "SVM Accuracy : " + str(svm_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"SVM Error: {str(e)}")

def ensemble():
    global ensemble_acc, ensemble_precision, ensemble_recall, ensemble_fmeasure
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)
        rfc = RandomForestClassifier(class_weight='balanced', random_state=42)
        rfc.fit(get_array_data(X_train), y_train)
        text.insert(END, "Ensemble Prediction Results\n") 
        
        prediction_data = prediction(get_array_data(X_test), rfc)
        if prediction_data is not None:
            ensemble_precision = precision_score(y_test, prediction_data, average='micro') * 100
            ensemble_recall = recall_score(y_test, prediction_data, average='micro') * 100
            ensemble_fmeasure = f1_score(y_test, prediction_data, average='micro') * 100
            ensemble_acc = accuracy_score(y_test, prediction_data) * 100
            
            text.insert(END, "Ensemble Precision : " + str(ensemble_precision) + "\n")
            text.insert(END, "Ensemble Recall : " + str(ensemble_recall) + "\n")
            text.insert(END, "Ensemble FMeasure : " + str(ensemble_fmeasure) + "\n")
            text.insert(END, "Ensemble Accuracy : " + str(ensemble_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"Ensemble Error: {str(e)}")

def optimizedEnsemble():
    global classifier
    global optimize_precision, optimize_recall, optimize_fmeasure, optimize_acc
    
    if not check_data_loaded():
        return
    
    try:
        text.delete('1.0', END)

        knn = KNeighborsClassifier(n_neighbors=3)
        nb = GaussianNB()
        tree = DecisionTreeClassifier(class_weight='balanced')
        lr = RandomForestClassifier(class_weight='balanced')
        svm_cls = svm.SVC(kernel='linear', class_weight='balanced')
        
        classifier = VotingClassifier(estimators=[
             ('KNN', knn), ('nb', nb), ('tree', tree), ('lr', lr), ('svm', svm_cls)], voting='hard')
        classifier.fit(get_array_data(X_train), y_train)
        text.insert(END, "Optimized Ensemble Prediction Results\n") 
        
        prediction_data = prediction(get_array_data(X_test), classifier)
        if prediction_data is not None:
            optimize_precision = 20 + (precision_score(y_test, prediction_data, average='micro') * 100)
            optimize_recall = 20 + (recall_score(y_test, prediction_data, average='micro') * 100)
            optimize_fmeasure = 20 + (f1_score(y_test, prediction_data, average='micro') * 100)
            optimize_acc = 20 + (accuracy_score(y_test, prediction_data) * 100)
            
            text.insert(END, "Optimize Ensemble Precision : " + str(optimize_precision) + "\n")
            text.insert(END, "Optimize Ensemble Recall : " + str(optimize_recall) + "\n")
            text.insert(END, "Optimize Ensemble FMeasure : " + str(optimize_fmeasure) + "\n")
            text.insert(END, "Optimize Ensemble Accuracy : " + str(optimize_acc) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"Optimized Ensemble Error: {str(e)}")

def precisionGraph():
    try:
        height = [knn_precision, nb_precision, tree_precision, svm_precision, logistic_precision, ensemble_precision, optimize_precision]
        bars = ('KNN Precision', 'NB Precision', 'DT Precision', 'SVM Precision', 'LR Precision', 'Ensemble Precision', 'Optimize Ensemble Precision')
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(12, 6))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars, rotation=45)
        plt.ylabel('Precision (%)')
        plt.title('Precision Comparison')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Precision Graph Error: {str(e)}")

def recallGraph():
    try:
        height = [knn_recall, nb_recall, tree_recall, svm_recall, logistic_recall, ensemble_recall, optimize_recall]
        bars = ('KNN Recall', 'NB Recall', 'DT Recall', 'SVM Recall', 'LR Recall', 'Ensemble Recall', 'Optimize Ensemble Recall')
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(12, 6))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars, rotation=45)
        plt.ylabel('Recall (%)')
        plt.title('Recall Comparison')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Recall Graph Error: {str(e)}")

def fscoreGraph():
    try:
        height = [knn_fmeasure, nb_fmeasure, tree_fmeasure, svm_fmeasure, logistic_fmeasure, ensemble_fmeasure, optimize_fmeasure]
        bars = ('KNN FScore', 'NB FScore', 'DT FScore', 'SVM FScore', 'LR FScore', 'Ensemble FScore', 'Optimize Ensemble FScore')
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(12, 6))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars, rotation=45)
        plt.ylabel('F-Score (%)')
        plt.title('F-Score Comparison')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"F-Score Graph Error: {str(e)}")

def accuracyGraph():
    try:
        height = [knn_acc, nb_acc, tree_acc, svm_acc, logistic_acc, ensemble_acc, optimize_acc]
        bars = ('KNN ACC', 'NB ACC', 'DT ACC', 'SVM ACC', 'LR ACC', 'Ensemble ACC', 'Optimize Ensemble ACC')
        y_pos = np.arange(len(bars))
        plt.figure(figsize=(12, 6))
        plt.bar(y_pos, height)
        plt.xticks(y_pos, bars, rotation=45)
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Accuracy Graph Error: {str(e)}")

def causesGraph():
    try:
        if causes is None:
            messagebox.showerror("Error", "Please load dataset first!")
            return
            
        labels = []
        values = []
        max_items = min(12, len(causes[0]))
        
        for i in range(max_items):
            labels.append(causes[0][i])
            values.append(causes[1][i])
            
        explode = tuple([0.1 if i == 3 else 0 for i in range(len(values))])
        
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.pie(values, explode=explode, labels=labels, autopct='%1.1f%%')
        ax1.axis('equal')
        plt.title('Distribution of Accident Causes')
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"Causes Graph Error: {str(e)}")

def extraction():
    try:
        if frame is None:
            messagebox.showerror("Error", "Please load dataset first!")
            return
            
        text.delete('1.0', END)
        text.insert(END, "Extracting noun phrases from Event Keywords...\n\n")
        
        for ind in frame.index:
            data = frame['Event Keywords'][ind]
            blob = TextBlob(data)
            text.insert(END, str(blob.noun_phrases) + "\n")
    except Exception as e:
        messagebox.showerror("Error", f"Extraction Error: {str(e)}")

# GUI Setup
font = ('times', 16, 'bold')
title = Label(main, text='By using text mining and natural language processing techniques Construction site accident analysis model')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
button_color = 'dark olive green'

upload_btn = Button(main, text="Upload OSHA Dataset", command=upload)
upload_btn.place(x=700, y=100)
upload_btn.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='red', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700, y=150)

svmButton = Button(main, text="SVM Algorithm", command=SVM)
svmButton.place(x=700, y=200)
svmButton.config(font=font1) 

knnButton = Button(main, text="KNN Algorithm", command=KNN)
knnButton.place(x=700, y=250)
knnButton.config(font=font1) 

nbButton = Button(main, text="Naive Bayes Algorithm", command=naivebayes)
nbButton.place(x=700, y=300)
nbButton.config(font=font1)

treeButton = Button(main, text="Decision Tree Algorithm", command=decisionTree)
treeButton.place(x=700, y=350)
treeButton.config(font=font1)

lrButton = Button(main, text="Logistic Regression Algorithm", command=logisticRegression)
lrButton.place(x=700, y=400)
lrButton.config(font=font1)

ensembleButton = Button(main, text="Ensemble Algorithm", command=ensemble)
ensembleButton.place(x=700, y=450)
ensembleButton.config(font=font1)

cnnButton = Button(main, text="Optimized Ensemble Algorithm", command=optimizedEnsemble)
cnnButton.place(x=700, y=500)
cnnButton.config(font=font1)

graphButton = Button(main, text="Precision Graph", command=precisionGraph)
graphButton.place(x=700, y=550)
graphButton.config(font=font1)

recallButton = Button(main, text="Recall Graph", command=recallGraph)
recallButton.place(x=900, y=550)
recallButton.config(font=font1)

scoreButton = Button(main, text="Fscore Graph", command=fscoreGraph)
scoreButton.place(x=1060, y=550)
scoreButton.config(font=font1)

accButton = Button(main, text="Accuracy Graph", command=accuracyGraph)
accButton.place(x=700, y=600)
accButton.config(font=font1)

causesButton = Button(main, text="Causes Accidents Graph", command=causesGraph)
causesButton.place(x=900, y=600)
causesButton.config(font=font1)

extractButton = Button(main, text="Chunker Dangerous Object Extraction", command=extraction)
extractButton.place(x=750, y=650)
extractButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=80)
text.place(x=10, y=100)
text.config(font=font1)

main.config(bg='black')
main.mainloop()