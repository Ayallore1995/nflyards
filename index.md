

# NFL Rushing Yards
Case Study based on NFL big Data Bowl 2022 conducted on Kaggle.

## Jump to Section:

[Introduction](#intro)<br>
[Performance Metrics](#performance-metrics)<br>
[Data](#data)<br>
[PreProcessing](#preprocessing)<br>
[Exploratory Data Analysis](#eda)<br>
[Feature Engineering](#feature-engineering)<br>
[Modelling](#modelling)<br>
[Deployment](#deployment)<br>
[Final Thoughts and Acknowledgments](#conclusion)<br>



## Intro
What is Football? Good question, even experts (on the internet) get quite heated hashing out the definition. For the purposes of this blog, we will refer to American Football as Football, the Hand-egg version. The excellent TV Show Friday Night Lights is based on this game. 

Basic rules of the game, 11 players on the field for each time. One team has possession and the other team tries to stop the offense from gaining ground. The offensive team has to gain 10 yards every 4 tries (called Downs). How do the teams achieve that? There are 2-3 ways, we will focus on Rushing plays. These account for more than 30% of the plays in the NFL. 
So, the basic gist of Rushing play is that one player called The Rusher is one who is tasked to carry the ball as far as possible. The remaining 10 offensive players look to clear a path for the Rusher. The defensive team looks to block or steal the ball from the rusher.
Here’s an example. It happens really fast. The defensive and offensive schemes are very complex. There’s a lot of trickery and bluffing involved to confuse the opponents (and invariably us in the process).


<p align="center">
  <img width="500" height="350" src="rush.gif">
</p>

 
### The Competition

The Kaggle competition was held by the NFL, they provided data for the seasons 2017 to 2019.
The data was divided into plays, each play given a unique ID, each play had the location, speed, acceleration, orientation, distance covered of all players involved, height, weight, position, so in total 22 player data for each play. And there were global features like Wind speed, weather, type of ground, temperature, humidity etc which were the same for all the players for a given play.
This data was recorded for just 0.1 secs. So not a lot!
As an armchair quarterback watching the game live may think they can predict how far the Rusher would go. The goal of the competition is to see what the data says and model it to predict How many Yards the Rusher would gain or lose.

### Why do this 
If we can find deeper insights using the given data, this could help coaches see if their schemes are working, if the players are in correct positions, what can be exploited, what can be improved.
It can also be used on tv to predict on live tv for the viewers, for post-game or pre-game shows breakdown of the games.  

So, let me summarise what we are doing. We are given 0.1 seconds of player information and we are expected to model it to find how many yards the rusher would go. Is it easy? Is it hard?
I don’t know, Let’s find out. 

### Business Constraints: 

1. Model Interpretability is important for the coaching decisions and checking its effectiveness.
2. Latency , if it is used on live tv as part of in-game graphics.

## Performance Metrics:

### 1. Continuous Ranked Probability Score.

CRPS was the metric of Evaluation for this Kaggle competition. It makes sense once you read about this metric.

Mathematically:  
 
![image](https://user-images.githubusercontent.com/77883553/161541788-a5af4f5c-8e52-4694-aa1a-fcebf2454cc7.png)

I'll admit it looks scary, It is not a common metric,
Let me explain some of the terms, first why 199? 199 is the number of buckets(classes) we have, on each play the team can either gain 99 yards, lose 99 yards or gain nothing(0th class) so thats 99+1+99= 199

***m=1 to N*** is just iterating over all the plays in the dataset,

***n=-99 to 99*** represents the yards lost to yards gained -99 representing 99 yards lost (or team loses the ball and the opponent gets a touchdown)  

***P(y<=n)***  represents the cumulative distribution function,

***H(n - Y_m))*** represents the the **Heaviside step function**, 

***H(x)  =1,x>=0***<br>
***H(x)  =0,x<0***<br>

let us see for 1 prediction how it would play out. 
Say our prediction is 5 yards will be gained. 

***Y_m=5*** in this case

so our CDF and Heaviside step function will look like

![image](https://user-images.githubusercontent.com/77883553/161543977-917be569-3a51-4008-b480-1588eea90c9d.png)

Another way to understand would be that it is the mean square error (MSE)of the predicted Cumulative density function (CDF) and the true Cumulative density function (CDF). 
The Continuous Ranked Probability Score (CRPS) generalizes the Mean Absolute Error (MAE) to the case of probabilistic forecasts. 

### 2. Mean Square error:

I tried to formulate it as a regression problem, but opted for CRPS since CRPS gave a way to interpret the output of the resultant model (MLP) as the discrete probability mass function and train to converge to our desired outputs.

As luck would have it, CRPS also gave better convergence compared to MSE. So a win-win in my books.



## Data:
There are 682154 rows of data, and 49 columns.
31007 unique plays across 3 seasons. 
Each row in the file corresponds to a single player's involvement in a single play.
The following features are included in the dataset:

### Player Specific features
-   X,Y Coordinates, speed, acceleration, Distance, Direction,Orientation, 
-   ID,Name, Jersey Number , height , weight ,Position

### Global Play features 
-   GameId,PlayId ,Home or Away Team, Season
-   YardLine, Quarter, PossessionTeam, Down, Distance
-   DefendersInTheBox, PlayDirection
-   **Yards** 
-   Position, StadiumType , Turf,  GameWeather

**Target Variable is *Yards***

## Research:
I went through the available Kaggle code solutions. The most impressive was the Winner’s solution. It used just 5 of the given variables (the relative distance and velocity)and used an ingenious method to win the competition. The rest of the competitors weren’t half bad! And I took a lot of inspiration and saw what worked and what didn’t. *Transfer Learning* if you may. 

## PreProcessing:
Let’s Explore, Dissect and Analyze the data (go along with my silly jokes). 

First, I load the data into a pandas dataframe. The csv file was around 240mb so not the biggest file, but reasonably big. 
One of the early steps is to make sure the data is complete, so I printed the columns causing the problems and the percentage to see how deep I would have to go to fix it.

<br>
{% gist b9fa6a019e3bbabf85ad79821817e6db %}

~~~python
Missing values in percentage
Orientation           0.003372
Dir                   0.004105
FieldPosition         1.261006
OffenseFormation      0.012900
DefendersInTheBox     0.003225
StadiumType           6.111523
GameWeather           8.820589
Temperature           9.330151
Humidity              0.903022
WindSpeed            13.467927
WindDirection        15.344922
dtype: float64
~~~

These are the steps I took to fix it. 
<br>
- **Orientation** and **Direction** of players filled with the mean of the orientation of all players based on their position like wingback, rusher etc.
- **Temperature** filled with the mean temperature based on the season (2017/18/19) and based on the week(1-11)
- Unknown **Field Positions** were given an 'unknown' value since it is a decent chunk of unknown values
- **Defenders in the box, OffenseFormation** and **Humidity** filled with  the median values
- **Stadium Type** was classed into open or closed only 
- **Game Weather** divided into four buckets, rainy, sunny, cold or overcast. Each bucket having it's own numerical value
- **Windspeed** filled with only numerical values extracted from the description
- **WindDirection** cleaned and assigned values in degrees instead of description, for example northeast given the value of 90 degrees.
<br>
[Link to code](https://gist.github.com/Ayallore1995/f2cd7de45c81cca69fc04c151491ff89)<br>
A snapshot of the code in the link <br>
![image](https://user-images.githubusercontent.com/77883553/161735789-81b4b82e-b5d1-4bba-883b-d1f31ee6bc93.png)<br>
As I noticed from the data and after doing research on the other solutions, I realised I had to standardise the directions, since sometimes the Yard lines, direction of players etc and location of play changed the data. 
My first question was can I even standardise the direction? And will it affect the Yards variable.
I plot the distribution of home and away and saw it basically had no difference
<br>
![image](https://user-images.githubusercontent.com/77883553/161704220-4b0cd086-6518-415b-8653-e8ce19038754.png)
<br>
Similarly, I standardised the X, Y and Yardline features. I checked first if changes would keep the distributions the same as before. No issues.<br>
![image](https://user-images.githubusercontent.com/77883553/161704340-75490b87-31a6-4c2b-96c2-ba45656b84ee.png)


### How the changes were made to standardise:
- Changing Direction of the plays to the Left if it is right. And switching the Coordinates, orientaion and other features dependent on the Direction of Play.
- Changing the possession team as always Home team and the defense team as away
- Creating a new feature Rusher to indicate if the player is the rusher or not
- Standardising the yardline
- Converting Direction from degrees to radians.
<br>
{% gist 58d3687ddf945e3ccdfa455b0776861b %}
	
<br>
### Odd data 
<br>
Fixing some data discrepancies:
2017 season had some discrepancies, one of them was in Distance feature, it didn’t match the distribution of the other 2 seasons. So, I manually adjusted to make it more along the lines of the other 2 years.   

Before and after 'fixing'
<br>
![image](https://user-images.githubusercontent.com/77883553/161726965-4ede6526-2bd7-47cf-8ebd-3ecd5588288b.png)

Another feature which didn’t align with the 2018 and 19 data was speed vs Distance correlation. As shown in the diagram, since the time was 0.1 seconds, speed should be roughly 10x distance but it was off, a simple substitution made it look nicer but didn’t improve the simple correlation, so I just let it be. 
 
![image](https://user-images.githubusercontent.com/77883553/161727150-68704b5c-1b21-4e59-baa4-367dbea2b989.png)

After making the substitution . 

![image](https://user-images.githubusercontent.com/77883553/161727194-a96cbf3c-3260-490c-83d3-0d0ef2d451b0.png)

Orientation of 2017 data was off by a phase of 90 degrees. So a simple fix.
	
![image](https://user-images.githubusercontent.com/77883553/161727258-a4e91e24-566f-444d-b13d-d697b39d6e81.png)


[Feature Engineering](#feature-engineering)<br>
[Modelling](#modelling)<br>
[Deployment](#deployment)<br>

## EDA
Now that the data is all cleaned up, Some basic stats 
<br>
![image](https://user-images.githubusercontent.com/77883553/161737593-e0b500eb-a1e8-43cd-b87e-5c6902b3a4a6.png)

The data was provided in between the 2019 season , so that explains the slightly less number of games from the 2019 season.

### Distribution of Yards
<br>
![image](https://user-images.githubusercontent.com/77883553/161725176-2c4ef6aa-3b9d-46f2-b938-516b8111c2e0.png)

### Percentile Values
I wanted to see the percentile values to see the values my potential model should be predicting.
As the table shows the values are concentrated around -1 and 10 (10th and 90th percentile)<br>
![image](https://user-images.githubusercontent.com/77883553/161737633-2dbe2d2f-1299-4006-b6eb-f0f539475d4e.png)

The InterQuartile range was just 5 yards.
### Downs vs Yards

Next I wanted to see if the Downs (Each try is called a down) had any effect on the Yards. So a boxplot for each Down. 
(If you’ve forgotten what a ‘Down’ is , each play has 4 tries and each try is called a Down)
	
![image](https://user-images.githubusercontent.com/77883553/161725635-8d931b09-c254-47f2-8580-55534a8104da.png)

First 3 downs mostly look alike but fourth Down the value is lower, It makes sense since the offense would rather kick the ball upfield and lose it instead of letting the opposition have an advantage.
This reflected in the central tendencies 
 
![image](https://user-images.githubusercontent.com/77883553/161726046-5d8fb325-a94a-4f6d-b09a-94c0f26d881a.png)

### Rusher Speed,Acceleration and Distance vs Yards

Another question I had was whether there was any relation between The Rusher speed, acceleration and Distance compared to Yards
	
![image](https://user-images.githubusercontent.com/77883553/161726096-6484da2b-82ba-48f7-be4e-cc71d248d264.png)

It looks very noisy with some possible relation, but nothing concrete.

### Rusher Play Position
	
Did the position of the Rusher matter?
Evidently so. The CB position yielded the best Yards gained.
	
![image](https://user-images.githubusercontent.com/77883553/161726156-3d20b5c2-e93c-4fa2-85be-8bd1012065bb.png)

### Defenders in Box vs Yards
Next was defenders in the box. Naturally the less defenders there were the more Yards were gained, nothing ground-breaking about this but nice to know I suppose.
	
![image](https://user-images.githubusercontent.com/77883553/161726251-6c4a2c33-7c46-4280-9a20-cf932b262044.png)

### Quarter vs Yards
The Quarter had basically no effect , other than overtime having a very small tail compared to others.  

![image](https://user-images.githubusercontent.com/77883553/161726400-ede347af-ab7a-4232-82cb-337d6ac059d7.png)

## Feature Engineering:

Next challenge was condensing 22 rows into 1 to feed into any model we want, I opted for having The Rusher as the frame of reference (Remember we made a new feature some sections back, this was why ). So, I found relative distance between all players vs the Rusher. I also added the inverse distance squared, this was taken from the research, inverse square worked better than other distances. 
{% gist eb4f0fd0785bb142ba4f49fbb011a4d8 %}<br>	
I created some new features, Speed and acceleration along the vertical and horizontal axis and found relative speeds along those axes. 
I also created a new feature called momentum (mass * velocity) thinking, momentum of a player could be helpful in breaking through lines or blocking the attacker from moving forward.
<br>
{% gist 5e849068de33c6c26b6c4f9d324bc171 %}<br>
	
One of the things that came up in my research was that this competition was similar to one that was held on Kaggle some time back, It was on molecule behaviour and predicting the movement, so this competition can be looked at in a similar way. Since both had "individual nodes" interacting with each other. 

Merged it with global play features. I did some feature selection and dropped the ones which weren’t contributing much like Turf, Player Height etc.

[Introduction](#intro)<br>
[Deployment](#deployment)<br>

## Modelling
I experimented with some models, I thought the models which could handle more complexities would fare better so I tried these.
 
### Decision Trees

{% gist e3ebc2e51d448076b698ce4630e45272 %}<br>
	
![image](https://user-images.githubusercontent.com/77883553/161723276-95c22e7d-e394-4da1-b878-0afabd304ecd.png)<br>

### Random Forest
	
{% gist 1cd8e257d986407c794583b69acbe81f %}<br>
	
![image](https://user-images.githubusercontent.com/77883553/161715186-e2a5f937-edf8-4a47-a484-5d4665fa0525.png)<br>

### XGBoost
{% gist 93e2049cc7af35894abfbec4e12ade99 %}<br>
	
![image](https://user-images.githubusercontent.com/77883553/161714768-55fd63a8-dd77-48ea-8300-5240f62ab910.png)<br>
	
###	Dense Neural Network

I used Tensorflow to construct and train the Neural Network.
I started small with just 1 or 2 hidden layers, ReLU activation, Reducing Learning rate every 3rd epoch.Early stopping was also used.
Used a custom function to measure CRPS , I used CRPS as the loss, I also experimented with Root Mean Square Error as loss, both were close in performance but ultimately crps just gave an edge.<br>
```python
def crps_nn(y_true,y_pred):
    loss = K.mean(K.sum((K.cumsum(y_pred, axis = 1) - K.cumsum(y_true, axis=1))**2, axis=1))/199
    return loss
```
Made sure to monitor Validation CRPS after each Epoch<br>
```python
class calls(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        y_pred = self.model.predict(X_test)
        y_true = np.clip(np.cumsum(Y_test, axis=1), 0, 1)
        y_pred = np.clip(np.cumsum(y_pred, axis=1), 0, 1)
        #y_pred=np.where(y_pred<0.1,0.0,1.0)
					     
        val_s = ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (199 * X_test.shape[0])
        print('val CRPS', val_s)
					  
```
Model consistently converged to a Validation CRPS of 0.0135(+-0.005) after 20 Epochs. It performed vastly better than the middling 0.22 from the above Machine Learning models. Even my initial smaller first try Neural Networks beat it comfortably.<br>

					   
### Model Summary
![image](https://user-images.githubusercontent.com/77883553/161713775-dc4552c2-c6fc-42f7-8feb-9d3737c22d99.png)<br>

### Why Neural Networks
Why did the Neural Network perform so well compared to the classic techniques?
					     
One possible explanation could be it creates these non linear features by itself through trial and error and just brute forces it way to the best of its ability.
The real world is complex, coming up with good features which may or may not work consumes a lot of time and effort. Neural Networks automate a lot of that process.

### Issues
I think it would be dangerous to just leave if the model works fine for now. It doesn't matter in this case because it is just football, a sport. But in medical and other high profile sectors which can impact a lot of people, I would be vary of releasing models without mapping out it's possible impacts. 	

## Deployment:
I began with Dask, a friend suggested it to me, I enjoyed it a lot , it had extensive and clear documentation. But I decided to go with Streamlit instead, as I wanted more functionality and a little more wiggle room. It also had great documentation and importantly gerat tutorials on Streamlit. I used Heroku to deploy my webpage built on streamlit. 

To make it interactive I converted the data into an image which showed a play at a time with player positions, direction of movement of each player and length of arrows indicated speed of player. So, using this visual information I ask the users to compete with my model. 

		
You can try it out here [Link to my webapp ](https://nflpredictiongame.herokuapp.com/)


![image](https://user-images.githubusercontent.com/77883553/161728042-fe58b82d-8b55-4505-892d-79ebe07d630d.png)<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/77883553/161728162-fd9c3660-e5c0-4191-95ca-d373111ef595.png">
</p>
<br>
<p align="center">
  <img src="https://user-images.githubusercontent.com/77883553/161728247-fc3e4a96-7ffe-4384-86d2-6bf8b72d3742.png">
</p>
<br>
I was very intimidated by all the talk about deployment for quite some time since I had no prior experince in Software Engineering, But the Internet is a wonderful place and I could figure out a way to piece information together to achieve this task.  
		
I had minimal experience  on how to deploy anything on Heroku , So I used the help of this video which made everything simple. Very well explained and everything worked without any issues. Highly recommend this video if you’re unsure how to deploy on Heroku. 
[Youtube Link](https://www.youtube.com/watch?v=nJHrSvYxzjE)

## Possible Improvements
-  Model calibration was something I wanted to include. 
-  2D- CNN architecture and transformer based methods which were used by the top Kaggle solutions
-  Data Augmentation to increase training data			
-  Graph Based Features and Graph Neural Networks
-  Dissect and figure out why and what parts of the Neural Network works so well.
-  Webpage improvements, Although I liked Streamlit, I wanted even more flexibility, I am excited to try new web dev frameworks.

## Conclusion
I had a great time doing this Case Study. Learning to do something for a specific task helped speed things up. Most of the issues I faced were to thread the needle through things I wanted to do, since everything is mostly available online. Learning to piece together information and making it cohesive would be my main takewaway from this project. 
		
## Future Projects
I would like to to try out the Next Competition held by the NFL in the future Super Bowl 2021, They provide a lot more data than 0.1 seconds worth. 
(Hope to fill this section soon)<br>
[Link to my Github](https://github.com/Ayallore1995)

## Contact
email: ayallore95@gmail.com<br>
[LinkedIn Profile](https://www.linkedin.com/in/abraham-ayallore-3a0011169/)
		
## Acknowledgements
I would like to thank the Applied AI team , who helped mentor this project and taught me real world applications of ML and DL.<br>
[NFL Big Bowl 2020](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/overview)<br>
[kaggle winner zoo et al](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400)<br>
[kaggle code by cpmpl](https://www.kaggle.com/code/cpmpml/initial-wrangling-voronoi-areas-in-python)<br>

		
## Revisit a section?:

[Back to the start](#intro)<br>



