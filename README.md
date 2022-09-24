# Biomedical-text-document-classification

## Dataset Link: https://www.kaggle.com/datasets/falgunipatel19/biomedical-text-publication-classification
For Biomedical text document classification, abstract and full papers(whose length less than or equal to 6 pages) available and used. This dataset focused on long research paper whose page size more than 6 pages. Dataset includes cancer documents to be classified into 3 categories like 'ThyroidCancer','ColonCancer','Lung_Cancer'. Total publications=7569. it has 3 class labels in dataset. number of samples in each categories: colon cancer=2579, lung cancer=2180, thyroid cancer=2810

### The LogistciRegression and DecisionTreeClassifier,RandomForestClassifer give the best result to the model.

# CONCLUSION
# About the data

#### In the data we use predict the Biomedical text document classification is wether it's Thyroid_Cancer,Lung_Cancer,Colon_Cancer based on the performed basicEDA, text preprocessing, build different models, such as LogisticRegression,DecisiontreeClassification,RandomForestClassication,XGBboostClassifier,For the above model Only All Algorithms have good accuracy score compare to the other model. After that we Run deeplearning model to the dataset. And create the Sequrentil model to the data and fit the data to the model in this model we use conv1d and several input layers used and finally we use 50 epochs to the model we get the 99% accurcy_score to the deep learning model.
