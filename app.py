import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from ipywidgets import interact
from sklearn.preprocessing import StandardScaler
import os
import random
import matplotlib
import plotly.express as px
from PIL import Image
from yellowbrick.cluster import KElbowVisualizer
from sklearn import set_config
set_config(display="diagram")
sns.set_theme(style="ticks", palette="pastel")


st.set_page_config(layout="wide")

#with st.sidebar:
# st.write("Hello")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
                .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 0.3rem;
                }
                .css-hxt7ib {
                    padding-top: 2rem;
                    padding-bottom: 1rem;
                }
                .st-bh {
                    background-color: white;
   
        </style>
        """, unsafe_allow_html=True)

    
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: blue;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #6F84FF;
    color:#ffffff
    }

</style>""", unsafe_allow_html=True) 
path = os.path.abspath(os.path.dirname(__file__))
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_model():
    model = joblib.load(os.path.join(path,"lgr_model.joblib"))
    return model
model = load_model()

@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_dataset ():
    data= pd.read_csv((os.path.join(path,"cleaned_file.csv")))
    return data
dataset = load_dataset()


st.header(" Employee Engagement Survey - Human Resources + AI")

# st.markdown("""
# <style>
# .big-font {
#     font-size:18px !important;
#     color :black;
# }
# </style>
# """, unsafe_allow_html=True)
# st.markdown('<p class="big-font"> With Machine Learning, HR can find meanigful insightful about their employees, as opposed to relying on obsolete theories and generalizations?', 
#     unsafe_allow_html=True)  

st.info(""" ###### This Machine Learning model is built to analyze and gain insights on a number of critical issues in 6 main areas obtained from employee feedback:
(1) Career Development
(2) Management Support
(3) Supervision & Working Environment
(4) Comp & Benefits
(5) Employee Engagement
(6) Peer Relationships        
            """ )

dataset_len = len(dataset)
questions = dataset.columns.value_counts().sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("# of employee surveyed", dataset_len)
col2.metric("Number of questions", questions-2)
col3.metric("Optimum Elbow(option to increase)", "5")
col4.metric("Data Source", "dataworld.com", "")



st.sidebar.header('Select desired number of cluster')
cluster_size = st.sidebar.radio("Clusters represent number of potential groupings", ([3,4,5]))
with st.sidebar.expander("Expand to view the Questions responded to by the employees"):
    columns =  [i for i in dataset.columns]
    st.write(columns[:39])

x= dataset.drop(['cluster'], axis =1)
        
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
scaled_x = sc.fit_transform(x)


# dimetionality reduction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_x= pca.fit_transform(scaled_x)
pca_x = pd.DataFrame(data = pca_x, columns=['PCA1', 'PCA2'])
       
from sklearn.cluster import KMeans
    
        
kmeans = KMeans(n_clusters= cluster_size)
kmeans.fit_transform(x)
pca_x_kmeans = pd.concat([pca_x,pd.DataFrame({'clusters' : kmeans.labels_})],axis = 1)

        
def get_colors ():
    if cluster_size == 2:
        s = ['red', 'blue']
    elif cluster_size == 3:
        s = ['red', 'blue','teal']
    elif cluster_size == 4:
        s= ['red', 'blue','teal','black']
    else:
        s =['red', 'blue','teal','black', 'orange']

    return s
color = get_colors()
            
survey_cluster = pd.concat([dataset, pd.DataFrame({'clusters' : kmeans.labels_})], axis = 1)       


#st.subheader('Data Explorations and visualization')

x = dataset.drop(['cluster'], axis =1)
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        #st.markdown(""" #####  Metric Selection""") 
        
        st.markdown("")
            
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def no_of_cluster ():
            
            fig, ax = plt.subplots(figsize =( 10, 5))
            model = KMeans()
            visualizer = KElbowVisualizer(model, k=(1,15)).fit(x)
            visualizer.show()
            return fig
        cluster_no = no_of_cluster()
        st.write(cluster_no)
        
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        def analysis():
            fig, ax = plt.subplots(figsize =( 10, 10))
            for c in survey_cluster.drop(['cluster','clusters'],axis =1).sample(axis =1):
                grid = sns.FacetGrid(data = survey_cluster, col='clusters')
                grid = grid.map(sns.histplot, c, )
                plt.show()
                return fig
        report = analysis()
        st.pyplot()
        
        #st.write([i for i in survey_cluster])
    

    with col2:            

        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        #def kmeans_plot ():
        def cluster_plot ():
            fig, ax = plt.subplots(figsize =( 13, 8))
            sns.scatterplot(data = pca_x_kmeans, x= 'PCA1', y='PCA2', palette = color, hue = 'clusters', ax= ax,s = 160)
            plt.grid(True)
            plt.show()
            plt.title("KMeans Cluster")
            return fig
        plot1 = cluster_plot()
        st.write(plot1)
        
      
        sns.set_theme(style="ticks", palette="pastel")
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        
        def cluster_counts ():
            fig, ax = plt.subplots(figsize =( 10, 6))
            sns.countplot(data = survey_cluster, x = 'clusters', palette= color)
            plt.title("Count of clusters by size")
            return fig
        plot2 = cluster_counts()
        st.write (plot2)
        



    
with st.sidebar:
    st.write("### Godwin Nwalozie")
    dir_name = os.path.abspath(os.path.dirname(__file__))
    file = Image.open(os.path.join(dir_name,"mazi_gunner2.jpg"))
    st.sidebar.image(file,width=200 )
    # Find me links
    kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
    st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
    st.sidebar.markdown(git,unsafe_allow_html=True)
    kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
    st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    
    st.sidebar.write(""" #### "Without Big Data, you are blind and deaf and in the middle of a freeway”, \
        Geoffrey Moore""")
    
        
        
          

            
     
    
