#CLASSIFICAZIONE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv("heart2.csv")
df.head()
print(df)
df.info()     #mostra informazioni su ciascuna colonna, come il tipo di dati e il numero di valori mancanti.
aa=df.isnull().sum()
print(aa)
for column in df:
    print(column)
    print(df[column].unique())
    cat_cols = ['sex', 'exng', 'cp', 'fbs', 'restecg']
    con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

# conversione dei tipi di colonna da numerici a categoriali
for column in df[cat_cols]:
    df[column] = df[column].astype('category')
df.info()

# converte le variabili categoriali in nomi per migliorare la leggibilità

df['sex'].replace({0:'Female', 1:'Male'}, inplace=True)
df['cp'].replace({0:'Asymptomatic', 1:'Typical Angina', 2:'Atypical Angina', 3:'Non-Anginal Pain'}, inplace=True)
df['fbs'].replace({0:'False', 1:'True'}, inplace=True)
df['restecg'].replace({0:'hypertrophy', 1:'normal', 2:'Abnormality'}, inplace=True)
df['exng'].replace({0:'No', 1:'Yes'}, inplace=True)
df[df.duplicated(keep=False)]
df.drop_duplicates(inplace=True)
df[df.duplicated(keep=False)]


###COLLEGAMENTO AL BROWSER
import webbrowser

url = 'report_eda.html'
webbrowser.open_new_tab(url)


sns.pairplot(df, kind="kde")
plt.show()

sns.pairplot(df, kind="kde", hue='output')
plt.show()

fig = plt.figure(figsize=(22,22))
gs  = fig.add_gridspec(4,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[2,0])
ax6 = fig.add_subplot(gs[2,1])
ax7 = fig.add_subplot(gs[3,0])
ax8 = fig.add_subplot(gs[3,1])

fig.suptitle(t='Counts of Categorical Columns by HeartDisease',y=0.94, fontweight ="bold",fontsize=22)
fig.set_facecolor("#F7F7F1")

# Sesso
ax1.set_title('Sex',fontweight ="bold",fontsize=15)
ax1.grid(color='#000000', linestyle='dashed', axis='x',dashes=(1,9))
sns.countplot(ax=ax1,data=df,x='sex', hue='output', fill=True, palette = ['blue','orange'])
ax1.legend([True, False])

# livello angina
ax2.set_title('Exercise Induced Angina',fontweight ="bold",fontsize=15)
ax2.grid(color='#000000', linestyle='dashed', axis='x',dashes=(1,9))
sns.countplot(ax=ax2,data=df,x='exng', hue='output', fill=True, palette = ['blue','orange'])
ax2.legend([True, False])

# tipo di dolore toracico
ax4.set_title('Chest Pain Type',fontweight ="bold",fontsize=15)
ax4.grid(color='#000000', linestyle='dashed', axis='x',dashes=(1,9))
sns.countplot(ax=ax4,data=df,x='cp', hue='output', fill=True, palette = ['blue','orange'])
ax4.legend([True, False])

# glicemia
ax5.set_title('Fasting Blood Sugar > 120 mg/dl',fontweight ="bold",fontsize=15)
ax5.grid(color='#000000', linestyle='dashed', axis='x',dashes=(1,9))
sns.countplot(ax=ax5,data=df,x='fbs', hue='output', fill=True, palette = ['blue','orange'])
ax5.legend([True, False])

# elettrocardiogramma
ax6.set_title('Resting Electrocardiographic Results',fontweight ="bold",fontsize=15)
ax6.grid(color='#000000', linestyle='dashed', axis='x',dashes=(1,9))
sns.countplot(ax=ax6,data=df,x='restecg', hue='output', fill=True, palette = ['blue','orange'])
ax6.legend([True, False])

plt.show()

sns.heatmap(df.corr(), annot=True, cmap="RdBu", center=0)
plt.show()

tabella=df[con_cols].describe().transpose()
print(tabella)

# codifica delle colonne categoriali
df_encoded = pd.get_dummies(df, columns = cat_cols, drop_first = True)


# definisco le caratteristiche e il target
X = df_encoded.drop(['output'],axis=1)
y = df_encoded[['output']]



# Train Test Split
from sklearn.model_selection import train_test_split

#split the data stratified  (esegue una suddivisione in modo che la proporzione dei valori nel campione prodotto sia uguale alla proporzione dei valori forniti al parametro)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 54, stratify=y)

# Models
from sklearn.neighbors import KNeighborsClassifier
#pipeline
from sklearn.pipeline import make_pipeline
# Scaling
from sklearn.preprocessing import StandardScaler


# Crea la variabile dei vicini da assegnare al modello da inserire come iperparametro
neighbors = np.arange(1, 15)

# crea dizionari per memorizzare i risultati
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

#create plot

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

knn = KNeighborsClassifier(n_neighbors=12)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Genera la matrice di confusione e il rapporto di classificazione
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
print(classification_report(y_test, y_pred))


# Crea la variabile dei vicini da assegnare al modello da inserire come iperparametro
neighbors = np.arange(1, 15)

# crea dizionari per memorizzare i risultati
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Set up a KNN Classifier in pipeline
    knn_std_mdl = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbor))

    # Fit the model
    knn_std_mdl.fit(X_train, y_train)

    # Compute accuracy
    train_accuracies[neighbor] = knn_std_mdl.score(X_train, y_train)
    test_accuracies[neighbor] = knn_std_mdl.score(X_test, y_test)

#create plot
# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()
print(max(test_accuracies, key=test_accuracies.get))

knn_std_mdl = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=12))

# Fit the classifier to the training data
knn_std_mdl.fit(X_train, y_train)

# Print the accuracy
print(knn_std_mdl.score(X_test, y_test))

##Metrics
from sklearn.metrics import classification_report, confusion_matrix
# Predict the labels of the test data: y_pred
y_pred = knn_std_mdl.predict(X_test)

# Genera la matrice di confusione e il rapporto di classificazione
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
print(classification_report(y_test, y_pred))

##REGRESSIONE LOGISTICA

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

# Import roc_curve
from sklearn.metrics import roc_curve

# Genera i valori della curva ROC: fpr, tpr, soglie
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Heart Attack Prediction')
plt.show()

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Calcola roc_auc_score
print("roc_auc_score: {}".format(roc_auc_score(y_test, y_pred_probs)))

# Print the accuracy
print("accuracy: {}".format(logreg.score(X_test, y_test)))

# Calcola la matrice di confusione
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
# Calcolare il rapporto di classificazione
print(classification_report(y_test, y_pred))

##CON LA STANDARDIZZAZIONE

# Instantiate the model
logreg_std = make_pipeline(StandardScaler(), LogisticRegression())

logreg_std.fit(X_train, y_train)
y_pred = logreg_std.predict(X_test)

# Calcola roc_auc_score
print("roc_auc_score: {}".format(roc_auc_score(y_test, y_pred_probs)))
# Print the accuracy
print("accuracy: {}".format(logreg_std.score(X_test, y_test)))

# Predict the labels of the test data: y_pred
y_pred = logreg_std.predict(X_test)

# Genera la matrice di confusione e il rapporto di classificazione
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()
print(classification_report(y_test, y_pred))




