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
                    padding-top: 1rem;
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

    
# button styling
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


st.header(" ¤(☆✭)¤ 𝐈𝐦𝐩𝐫𝐨𝐯𝐢𝐧𝐠 𝐄𝐦𝐩𝐥𝐨𝐲𝐞𝐞 𝐄𝐧𝐠𝐚𝐠𝐞𝐦𝐞𝐦𝐞𝐧𝐭 𝐰𝐢𝐭𝐡 𝐀𝐈-𝐌𝐋 ¤(✭☆)¤")
st.write(" ##### Machine Learning Model : by Godwin Nwalozie")

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
st.warning ("""  𝖢𝖺𝗋𝖾𝖾𝗋 𝖣𝖾𝗏𝖾𝗅𝗈𝗉𝗆𝖾𝗇𝗍 ♦♦ 𝖬𝖺𝗇𝖺𝗀𝖾𝗆𝖾𝗇𝗍 𝖲𝗎𝗉𝗉𝗈𝗋𝗍 ♦♦  𝖲𝗎𝗉𝖾𝗋𝗏𝗂𝗌𝗂𝗈𝗇 & 𝖶𝗈𝗋𝗄𝗂𝗇𝗀 𝖤𝗇𝗏𝗂𝗋𝗈𝗇𝗆𝖾𝗇𝗍 ♦♦ 𝖢𝗈𝗆𝗉 & 𝖡𝖾𝗇𝖾𝖿𝗂𝗍𝗌 ♦♦ 𝖤𝗆𝗉𝗅𝗈𝗒𝖾𝖾 𝖤𝗇𝗀𝖺𝗀𝖾𝗆𝖾𝗇𝗍 ♦♦ 𝖯𝖾𝖾𝗋 𝖱𝖾𝗅𝖺𝗍𝗂𝗈𝗇𝗌𝗁𝗂𝗉𝗌""" )

dataset_len = len(dataset)
questions = dataset.columns.value_counts().sum()
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of employee surveyed", dataset_len)
    col2.metric("Number of questions", questions-2)
    col3.metric("Clustering Algorithm", "KMeans")
    col4.metric("Model Prediction Accuracy Score", "95%", "")


st.sidebar.title('Select desired number of cluster')
from sklearn.cluster import KMeans
cluster_size = st.sidebar.radio("A cluster is a group based on characteristics or similar attributes", ([3,4,5]))


x= dataset.drop(['cluster'], axis =1)






from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
scaled_x = sc.fit_transform(x)


# dimetionality reduction with PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_x= pca.fit_transform(scaled_x)
pca_scaled_x = pd.DataFrame(data = pca_x, columns=['PCA1', 'PCA2'])


kmeans = KMeans(n_clusters= cluster_size)
kmeans.fit_transform(pca_scaled_x)
pca_x_kmeans = pd.concat([pca_scaled_x,pd.DataFrame({'clusters' : kmeans.labels_})],axis = 1)

survey_cluster = pd.concat([dataset, pd.DataFrame({'clusters' : kmeans.labels_})], axis = 1)



dataset_with_label = pd.concat([dataset, pd.DataFrame({'clusters' : kmeans.labels_})], axis = 1)  


st.sidebar.download_button("Download the training dataset)", data =dataset_with_label .to_csv(), file_name = 'engagement_survey.csv', mime ="text/csv")
st.sidebar.download_button("Download the trained model(joblib)", b'lgr_model',file_name = 'model.joblib')


with st.sidebar.expander("Expand to view the Questions responded to by the employees"):
    columns =  [i for i in dataset.columns]
    st.write(columns[:39])


def get_colors ():
    if cluster_size == 2:
        s = ['green', 'red']
    elif cluster_size == 3:
        s = ['green', 'red','blue']
    elif cluster_size == 4:
        s= ['green', 'red','blue','orange']
    else:
        s =['green', 'red','blue','orange', 'indigo']

    return s
color = get_colors()
            
     


#st.subheader('Data Explorations and visualization')

x = dataset.drop(['cluster'], axis =1)
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        #st.markdown(""" #####  Metric Selection""") 
        
        st.markdown("")
        
        st.write(" ##### YellowBrick - KElbowVisualizer")  
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def no_of_cluster ():
            fig, ax = plt.subplots(figsize =(8, 5))
            model = KMeans()
            visualizer = KElbowVisualizer(model, k=(1,15)).fit(x)
            visualizer.show()
            return fig
        cluster_no = no_of_cluster()
        st.write(cluster_no)
        
    
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        def analysis():
            st.write(" ##### Random generated reports - based on selected cluster size")
            fig, ax = plt.subplots(figsize =(8, 10))
            for c in survey_cluster.drop(['clusters','cluster'],axis =1).sample(axis =1):
                grid = sns.FacetGrid(data = survey_cluster, col='clusters')
                grid = grid.map(sns.histplot, c )
                plt.show()
                return fig
        report = analysis()
        st.pyplot()
        
        st.write("Model")
        st.write(model)
        
    

    with col2:            
        st.markdown("")
        st.write(" ##### KMeans Cluster") 
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def cluster_plot ():
            fig, ax = plt.subplots(figsize =( 10, 8))
            sns.scatterplot(data = pca_x_kmeans, x= 'PCA1', y='PCA2', palette = color, hue = 'clusters', ax= ax,s = 200)
            ax =sns.scatterplot(x = kmeans.cluster_centers_[:, 0], y= kmeans.cluster_centers_[:, 1], 
                hue= range(cluster_size), palette=color, s=200, ec='black', marker = "*",  legend = False,ax=ax)
            plt.title("Here you can see employees are grouped based on similar attributes or feedback", fontsize =16)
            plt.show()
            
            return fig
        plot1 = cluster_plot()
        st.write(plot1)
        
      
        sns.set_theme(style="ticks", palette="pastel")
        @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        
        def cluster_counts ():
            fig, ax = plt.subplots(figsize =( 12, 5))
            sns.countplot(data = survey_cluster, x = 'cluster', palette= color, saturation= 0.75)
            plt.title("Count of clusters by size")
            return fig
        plot2 = cluster_counts()
        st.write (plot2)
        



    
with st.sidebar:
    st.write("### Godwin Nwalozie")
    dir_name = os.path.abspath(os.path.dirname(__file__))
    # file = Image.open(os.path.join(dir_name,"mazi.png"))
    # st.sidebar.image(file,width= 300 )
    # Find me links
    kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
    st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
    st.sidebar.markdown(git,unsafe_allow_html=True)
    kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
    st.sidebar.markdown(kaggle,unsafe_allow_html=True)
    
    st.sidebar.write(""" #### "Without Big Data, you are blind and deaf and in the middle of a freeway”, \
        Geoffrey Moore""")
    
        
        
          

            
     
    
