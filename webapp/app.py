import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import pandas as pd
import numpy as np
import re
from string import punctuation
import math
import tensorflow.keras 
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import time
from io import BytesIO



st.set_page_config(layout="wide")

cols=st.columns(3)
with cols[1]:
    st.title("NFL Prediction Game!")
st.subheader("How the game works")
st.subheader("I will show you a diagram of player positions, with arrows representing Directions, length of arrows representing Speed, the Blue represents the attacking Team, the Orange represents the defending team")
st.subheader(r"The Red colour indicating The Rusher(The Player with the Football)")
st.subheader("The data is collected after 0.1 seconds after start of play")
st.subheader("You need to guess how many Yards will be gained or lost just from this 0.1 seconds of information (visual for you)")
st.subheader("Let's see how you compare with my Model!")
st.write("Guess between -1 and 10 , just a hint")

# Load data
final_df=pd.read_csv('Final_Best_data.csv')
train=pd.read_csv('Bestresulting.csv')
#Removing 2017 data, data is better collected later on
#train=train.loc[train.Season!=2017,['PlayId','X','Y','Team','S','Dis','PlayDirection','Dir','NflId','NflIdRusher','Yards','Season']]
final_df.set_index('PlayId',inplace=True)
final_df.drop("Unnamed: 0",inplace=True,axis=1)

#base function from https://www.kaggle.com/code/robikscube/nfl-big-data-bowl-plotting-player-position/notebook
def create_football_field(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(20, 10)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='white', zorder=0)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='black')
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')
    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='green',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='green',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='black')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='black', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='black')
        ax.plot([x, x], [53.0, 52.5], color='black')
        ax.plot([x, x], [22.91, 23.57], color='black')
        ax.plot([x, x], [29.73, 30.39], color='black')

    if highlight_line:
        hl = highlight_line_number + 10
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
    return fig, ax

def predict_one(id,final_df):
    cols=st.columns(3)
    
    with cols[1]:
        int_val=st.number_input("Enter Prediction Here" ,min_value=-99, max_value=99, value=0, step=1)

    Y=final_df.Yards
    Y=np.array(Y).reshape(-1,1)

    targets = Y

    test=np.zeros((Y.shape[0],199))
    for t in list(Y):
        test[0][99 + int(t[0])] = 1
    final_df.drop('Yards',inplace=True)

    def crps_nn(y_true,y_pred):
            loss = K.mean(K.sum((K.cumsum(y_pred, axis = 1) - K.cumsum(y_true, axis=1))**2, axis=1))/199
            return loss

    model=load_model('nfl_pred.hdf5',custom_objects={'crps_nn':crps_nn})   

    final_df=np.array(final_df).reshape(1,-1)
    preds=model.predict(final_df)

    n=10
    top5=np.argsort(preds,axis=1)[0][::-1]
    top5_prob=np.sort(preds,axis=1)[0][::-1]
    top_n=np.zeros((n,2))
    for i in range(n):
        top_n[i,0]=top5[i]
        top_n[i,1]=np.round(top5_prob[i]*100,2)
    fig,ax=plt.subplots()

    sns.barplot(x=top_n[:,0].astype('int')-99,y=top_n[:,1],ax=ax,palette='viridis')
    plt.ylabel('Probabilities')
    plt.xlabel('Yards')


    cols=st.columns(3)
    with cols[1]:           

        with st.expander('Click to Reveal Answer',expanded=False):
            if int_val==targets:
                st.write("Congrats you did well!")
            else:
                st.write("Just off by {0} Yards".format(abs(int(targets[0][0]-int_val))))
            
            st.subheader('Answer was {0} Yards'.format(np.argmax(test)-99))
        with st.expander('Click to see how the Model did',expanded=False):

            st.subheader('Ground Truth {0} Yards,Model Predicted with {1}% probability'.format(np.argmax(test)-99,round(preds[0][np.argmax(test)]*100,2)))

            ax.bar_label(ax.containers[0])

            buf2 = BytesIO()
            fig.savefig(buf2, format="png")
            st.image(buf2)
            st.write("Here are the top 10 prediction of the model based on probabilities" )

id=train.sample(1).PlayId.values[0]
final_df=final_df.loc[id]

def print_image(id):
    
    fig, ax = create_football_field()
    play=train.loc[train.PlayId==id]
    rush=play[play.NflId==play.NflIdRusher]
    play=play[play.NflId!=play.NflIdRusher]
    plt.scatter(x=play.loc[play.Team=='home','X'],
                y=play.loc[play.Team=='home','Y'],
                color='blue',s=200,marker='o',edgecolor='white')
    for i,j in play.loc[play.Team=='home'].iterrows():
        plt.arrow(j.X,j.Y,
                  dx=np.cos(j.Dir)*np.pi/180.0,
                  dy=np.sin(j.Dir)*np.pi/180.0,head_width=0.8,
                  head_length=4,fc='blue',ec='blue')

    plt.scatter(x=rush.X,y=rush.Y, marker='o',color='red',s=1000)
    plt.arrow(x=rush.X.values[0],y=rush.Y.values[0],
              dx=np.cos(rush.Dir.values[0])*np.pi/180.0,
              dy=np.sin(rush.Dir.values[0])*np.pi /180.0,
              head_width=0.8,head_length=j.S,
              fc='red',ec='black')

    plt.scatter(x=play.loc[play.Team=='away','X'],y=play.loc[play.Team=='away','Y'],color='orange',s=200,marker='o',edgecolor='white')
    for i,j in play.loc[play.Team=='away'].iterrows():
        plt.arrow(j.X,j.Y,
                  dx=np.cos(j.Dir)*np.pi/180.0,
                  dy=np.sin(j.Dir)*np.pi/180.0,head_width=0.8,
                  head_length=4,fc='orange',ec='orange')
    plt.title('Play Direction')
    direction={'left':'<|-','right':'-|>'}
    plt.annotate(text='', xy=(65,57), 
                 xytext=(55,57), 
                 arrowprops=dict(color='black',linewidth=3,mutation_scale=10,arrowstyle=direction[play.PlayDirection.values[0]]))
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

if 'id' not in st.session_state:
    st.session_state['id'] = id

print_image(st.session_state.id)
predict_one(st.session_state.id,final_df)


st.header("FAQs:")
st.subheader("""
    Question : Where is this data from?
     """)
with st.expander("Click to expand"):
    st.write("Data is originally from the 2020 NFL Big Bowl challenge on Kaggle")
    st.markdown("""---""")
st.subheader("""
    Question : What type of Model is it?
     """)
with st.expander("Click to expand"):
    st.write("Its a simple Dense Neural Network, the basic boring kind, I would like to improve it and try to do a 2-D CNN based model, The kaggle winner solution was based on that")    
st.markdown("""---""")
st.subheader("""
    Question : How did classic Machine Learning Techniques fare?
     """)
with st.expander("Click to expand"):
    st.write("Not good!(enough), they were ok at best, The Metric used was CRPS (Continuous Ranked Probability Score), The lower the better , The best model gave me a score of around 0.013, whereas the classical techniques gave about 0.22 at best")    
st.markdown("""---""")
st.subheader("""
    Question : Wait hold on, What's a 'CRPS'?
     """)
with st.expander("Click to expand"):
    st.write("It's not the most well known metric. It a probabilistic metric, it's a way to compare cumulative distribution function (CDF) of the ground truth vs predictions, It takes the absolute mean error of the cdfs")    
st.markdown("""---""")
st.subheader("""
    Question : Is this the full data'?
     """)
with st.expander("Click to expand"):
    st.write("No, I'm only loading the plays which my model predicted correctly with the best separation in predictions,Also helps reduce the space to run this page")