# NFL Rushing Yards
Case Study based on NFL big Data Bowl 2022 conducted on Kaggle.
## Intro
What is Football? Good question, even experts (on the internet) get quite heated hashing out the definition. For the purposes of this blog, we will refer to American Football as Football, the Hand-egg version. The excellent TV Show Friday Night Lights is based on this game. 

Basic rules of the game, 11 players on the field for each time. One team has possession and the other team tries to stop the offense from gaining ground. The offensive team has to gain 10 yards every 4 tries (called Downs). How do the teams achieve that? There are 2-3 ways, we will focus on Rushing plays. These account for more than 30% of the plays in the NFL. 
So, the basic gist of Rushing play is that one player called The Rusher is one who is tasked to carry the ball as far as possible. The remaining 10 offensive players look to clear a path for the Rusher. The defensive team looks to block or steal the ball from the rusher.
Here’s an example. It happens really fast. The defensive and offensive schemes are very complex. There’s a lot of trickery and bluffing involved to confuse the opponents (and invariably us in the process).
 
## The Competition

The Kaggle competition was held by NFL, they provided data for the seasons 2017 to 2019.
The data was divided into plays, each play given a unique ID, each play had the location, speed, acceleration, orientation, distance covered of all players involved, height, weight, position, so in total 22 player data for each play. And there were global features like Wind speed, weather, type of ground, temperature, humidity etc which were same for all the players for a given play.
This data was recorded for just 0.1 secs. So not a lot!
As an armchair quarterback watching the game live may think they can predict how far the Rusher would go. The goal of the competition is to see what the data says and model it to predict How many Yards the Rusher would gain or lose.

## Why do this 
If we can find deeper insights using the given data, this could help coaches see if their schemes are working, if the players are in correct positions, what can be exploited, what can be improved.
It can also be used on tv to predict on live tv for the viewers, for post-game or pre-game shows breakdown of the games.  

So, let me summarise what we are doing. We are given 0.1 seconds of player information and we are expected to model it to find how many yards the rusher would go. Is it easy? Is it hard?
I don’t know, Let’s find out. 

## Performance measure:
### 1. CRPS or Continuous Ranked Probability Score.

 Mathematically:  
 
 ![image](https://user-images.githubusercontent.com/77883553/161541788-a5af4f5c-8e52-4694-aa1a-fcebf2454cc7.png)

I'll admit it looks scary, It is not a common metric,
Let me explain some of the terms, first why 199? 199 is the number of buckets(classes) we have, on each play the team can eihter gain 99 yards, lose 99 yards or gain nothing so thats 99+1+99= 199

***m=1 to N*** is just iterating over all the plays in the dataset,

***n=-99 to 99*** represents the yards lost to yards gained -99 representing 99 yards lost (or team loses the ball and the opponent gets a touchdown)  

***P(y<=n)***  represents the cummulative distributive function,

***H(n - Y_m))*** represents the the Heaviside step function, 

***H(x)=1 for x>=0 and 0 otherwise***

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


## Possible Issues: 

1. Model Interpretability is important for the coaching decisions and checking it’s effectiveness.
2. Latency , if it is used on live tv as part of the graphics.

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

**Target Variable is *Yards* **

## Research:
I went through the available Kaggle code solutions. The most impressive was the Winner’s solution. It used just 5 of the given variables (the relative distance and velocity)and used an ingenious method to win the competition. The rest of the competitors weren’t half bad! And I took a lot of inspiration and saw what worked and what didn’t. *Transfer Learning* if you may. 

## EDA:
Let’s Explore, Dissect and Analyse the data (go along with my silly jokes). 

First, I load the data into a pandas dataframe. The csv file was around 240mb so not the biggest file, but reasonably big. 
One of the early steps is to make sure the data is complete, so I printed the columns causing the problems and the percentage to see how deep I would have to go to fix it.


~~~python

in: print("Missing values in percentage\n ",data.isnull().sum()[data.isnull().sum()>0]/len(data)*100)
out: 
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
 
As I noticed from the data and after doing research on the other solutions, I realised I had to standardise the directions, since sometimes the Yard lines, direction of players etc and location of play changed the data. 
My first question was can I even standardise the direction? And will it affect the Yards variable.
I plot the distribution of home and away and saw it basically had no difference

<object 
    width="100%"
    height="350"    
    src="data:text/html;charset=utf-8,
    <head><base target='_blank' /></head>
    <body><script src='https://gist.github.com/f2cd7de45c81cca69fc04c151491ff89.git'></script>
    </body>">

Similarly, I standardised the X, Y and Yardline features. I checked first if changes would keep the distributions same as before. No issues.
 

Changes made to standardise:
- Changing Direction of the plays to the Left if it is right. And switching the Coordinates, orientaion and other features dependent on the Direction of Play.
- Changing the possession team as always Home team and the defense team as away
- Creating a new feature Rusher to indicate if the player is the rusher or not
- Standardising the yardline
- Converting Direction from degrees to radians.
The code is given below 

 

Now the data is all cleaned up and looking handsome, let’s see the basic stats 



 

The data was provided in between the 2019 season , so that explains the slightly less number of games from 2019 season.

I wanted to see the percentile values to see the values my potential model should be predicting.
As the table shows the values are concentrated around -1 and 10 (1oth and 90th percentile)
 

The Inter Quartile range was just 5 yards.

Next I wanted to see if the Downs had any effect on the Yards. So a boxplot for each Down. 
(If you’ve forgotten what a ‘Down’ is , each play has 4 tries and each try is called a Down)
 
First 3 downs mostly look alike but fourth Down the value is lower, It makes sense since the offense would rather kick the ball upfield and lose it instead of letting the opposition have an advantage.
This reflected in the central tendencies 
 





Another question I had was whether there was any relation between The Rusher speed, acceleration and Distance compared to Yards
 

It looks very noisy with some possible relation, but nothing concrete.


Did the position of the Rusher matter?
Evidently so. The CB position yielded the best Yards gained.
 


Next was defenders in the box. Naturally the less defenders there were the more Yards were gained, nothing ground-breaking about this but nice to know I suppose.
 
The Quarter had basically no effect , other than overtime having a very small tail compared to others.  





Fixing some data discrepancies: 
2017 season had some discrepancies, one of them was in Distance feature, it didn’t match the distribution of the other 2 seasons. So, I manually adjusted to make it more along the lines of the other 2 years.   

 
 






Another feature which didn’t align with the 2018 and 19 data was speed vs Distance correlation. As shown in the diagram, since the time was 0.1 seconds, speed should be roughly 10x distance but it was off, a simple substiution made it look nicer but didn’t improve the simple correlation, so I just let it be. 
 
After making the substitution. 

 








Orientation of 2017 data was off by a phase of 90 degrees. So a simple fix.
 


Feature Engineering:
Next challenge was condensing 22 rows into 1 to feed into any model we want, I opted for having The Rusher as the frame of reference (Remember we made a new feature some paragraphs back, this was why ). So, I found relative distance between all players vs the rusher. I also added the inverse distance squared, this was taken from the research, inverse square worked better than other distances. 
I created some new features, Speed and acceleration along the vertical and horizontal axis and found relative speeds along those axes. 
I also created a new feature called momentum(mass * velocity), One of the things that came up in my research was that this competition was similar to one that was held on Kaggle some time back, It was on molecule behaviour and predicting the movement, so this competition can be looked at in a similar way. 
   
Finally merged it with global play features. I did some feature selection and dropped the ones which weren’t contributing much.

Models I tried:
Conducted Gridsearch CV on the machine learning models. These were the outcomes.
1.	XGBoost
 
2.	Random Forest
 
3.	Decision Trees
 
4.	Dense Neural Network
This is the architecture after trail and error. Tried different learning rates and different number of layers and nodes, but all were vastly outperforming the above algorithms
 
After 100 epochs, it had a validation CRPS of around 0.0135, vastly better than the middling 0.22 from the above models.

	 


Summary of the model performance.
 

Deployment:
I used streamlit to build the website and Heroku to deploy it. I first started with Dask which was easier but I wanted a bit more functionality to so I switched to Streamlit. It had very good documentation and I had the world’s best instructor YouTube to help me out whenever I got stuck.

To make it interactive I converted the data into an image which showed a play at a time with player positions, direction of movement of each player and length of arrows indicated speed of player. So, using this visual information I ask the users to compete with my model. 
    

You can try it out here [https://nflpredictiongame.herokuapp.com/]
I had minimal experience on how to deploy anything on Heroku , So I used the help of this video which made everything super simple. Very well explained and everything worked without any issues. Highly recommend this video if you’re unsure how to deploy on Heroku. 
[https://www.youtube.com/watch?v=nJHrSvYxzjE]

