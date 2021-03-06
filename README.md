# About the project

This project was part of course [**CSE258 Web Mining and Recommender Systems**](http://cseweb.ucsd.edu/classes/fa17/cse258-a/) at UC San Diego by Prof. Julian McAuley.
It used restaurant reviews in Paris to predict two tasks: </br>
    1. Visit Prediction - given a pair of (user, business), predict whether the user visited the business before. In other words, whether the user reviewed that business. <br>
    2. Rating Prediction - given a pair of (user, business), predict the star rating that a user will give to that particular business.
   
The model evaluations are done on Kaggle internal competitions. 

# Credits
This project used scikit-learn to build model pipelines. 
The structure and framework used in this repository was inspired by [Josh Montague's article](http://joshmontague.com/posts/2016/mnist-scikit-learn/) on how to build a framework for Kaggle competition that will allow fast iterations on building, selecting and evaluating the models. </br></br>
In fact, part of the code came from an adaptation of his repository: </br> 
**MNIST + scikit-learn** https://github.com/jrmontag/mnist-sklearn 
</br>
</br>
I also used DataframeMapper from sklearn-pandas to help bridge pandas DF to work with sklearn Pipeline. Helps a lot! </br>
**sklearn-pandas** https://github.com/scikit-learn-contrib/sklearn-pandas
</br>
</br>
I acknowledge and am grateful for their contribution and knowledge sharing.
