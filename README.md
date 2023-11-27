# Galaxy-Morphological-Classification.
Galaxy Morphological Classification means classifying galaxies according to there shapes and sizes.
So, For galaxy morphological classification we took dataset "Galaxy Zoo" from wwww.kaggle.com.
We have laballed data according to the main three types of Galaxies Spiral , Elipitical , Lenticular.
We have used CNN deep learing model for this because CNN is an deep learning better for working on Images as well as for Classsifications.
Why classifying Galaxies there most be some reason to Classify Galaxies?
An Astophysicist needs to Classify Galaxies in order to find the Galactic Evolution, Probing structure of universe, Formation of Stars, Predicting Cosmic Evolution ,Finding Dark Matter and etc.
Classifying Galaxies is an time consuming task for Astrophysicist and Astronomers so to safe alot of time. I have created an AI model which can predict the main three types of galaxies. In Future will increase the classes of Galaxies as the sub types of all the other galaxies.   
It's my contribution to AI community and Astrophysicist Community.
So, In this model first I have Cleaned data and filtered out the labels for these 3 main types of Galaxies (Requires domain knowledge for it).
Then Some Advanced level EDA(Exploratory Data Analysis)
Applied Augmentation to increase our train_data and then Applied CNN model from scratch. Training model on 100 epouchs while using categorical_cross_entropy as loss function and Optimizer "Adam" while metric "Accuracy"
After training our Model got an amazing results of 84% Accuracy.
