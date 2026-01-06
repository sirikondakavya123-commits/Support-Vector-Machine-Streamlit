import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

#logger
def log(message):
    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
#session state initialization
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False


#folder setup
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"data","raw")
CLEANED_DIR=os.path.join(BASE_DIR,"data","cleaned")
os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEANED_DIR,exist_ok=True)

log("Application started.")
log(f"RAW_DIR = {RAW_DIR}")
log(f"CLEANED_DIR = {CLEANED_DIR}")

#page config

st.set_page_config(page_title="End-TO-End SVM",layout="wide")
st.title("End-to-End SVM Platform")

#Sidebar : Model settings
st.sidebar.header("SVM settings")

kernal=st.sidebar.selectbox("kernal",options=["linear","poly","rbf","sigmoid"])
c=st.sidebar.slider("C (Regularization)",0.01,10.0,1.0)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])
log(f"SVM settings ---> kernal = {kernal}, c={c}, Gamma={gamma}")

#step 1:Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1: Data Ingestion started.")

option=st.radio("Choose Data Source",["Download Dataset","Upload CSV"])
df=None
raw_path=None
if option=="Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris dataset.")
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response=requests.get(url)
        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)
        df=pd.read_csv(raw_path)
        st.success("Dataset downloaded successfully.")
        log(f"Iris dataset saved at {raw_path}.")
if option=="Upload CSV":
    uploaded_file=st.file_uploader("Upload CSV File",type=["csv"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read(raw_path)
        st.success("File uploaded successfully.")
        log(f"Uploaded data saved at {raw_path}.")
#step 2:EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2: EDA started....")
    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Missing values",df.isnull().sum())
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True,ax=ax)
    st.pyplot(fig)
    log("EDA completed.")
#step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")
    strategt=st.selectbox(
        "Missing value strategy",
        ["Mean","Median","Drop Rows"]
    )
    df_clean=df.copy()
    if strategt=="Drop Rows":
        df_clean=df._clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=np.number):
            if strategt=="Mean":
                df_clean[col]=df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col]=df_clean[col].fillna(df_clean[col].median())
    st.session_state.df_clean=df_clean
    st.success("Data cleaning completed.")
else:
    st.info("Please complete Step 1 (Data Ingestion) First....")
#Step 4:Save the cleaned data
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data found. Please complete step 3 first.")
    else:
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename=f"cleaned_dataset_{timestamp}.csv"
        clean_path=os.path.join(CLEANED_DIR,clean_filename)

        st.session_state.df_clean.to_csv(clean_path,index=False)
        st.success("cleaned dataset saved.")
        st.info(f"saved at : {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}.")
#step 5:Load cleaned dataset
st.header("step 5: Load Cleaned Dataset")
clean_file=os.listdir(CLEANED_DIR)
if not clean_file:
    st.warning("No cleaned datasets found. Please save one in Step 4.")
    log("No cleaned dataset available")
else:
    selected=st.selectbox("Select Cleaned Dataset",clean_file)
    df_model=pd.read_csv(os.path.join(CLEANED_DIR,selected))
    st.success(f"Loaded dataset :{selected}")
    log(f"Loaded cleaned dataset : {selected}")
    st.dataframe(df_model.head())

#Step 6 : Train SVM
st.header("Step 6 : Train SVM")
log("Step 6 started: SVM Training")
target=st.selectbox("select Target Column",df_model.columns)
y=df_model[target]
if y.dtype=="object":
    y=LabelEncoder().fit_transform(y)
    log("Target Column encoded")
#Select numeric features only
x=df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)
if x.empty:
    st.error("No Numeric features available for the training..")
    st.stop()
#Sclae features
scaler=StandardScaler()
x=scaler.fit_transform(x)
#Train-Test-Split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.25,random_state=42)
#SVM
model=SVC(kernel=kernal,C=c,gamma=gamma)
model.fit(x_train,y_train)
#Evaluate
y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)
st.success(f"Accuracy : {acc:.2f}")
log(f"SVM trained successfully |Accuracy={acc:.2f}")
cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
st.pyplot(fig)
