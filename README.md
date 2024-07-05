# Project description
The project involves data collection, analysis, visualization, and machine learning algorithm implementation focused on book data.

## Tasks Overview
### Data Collection
Implemented a web crawler and scraper to collect book data from the following website: https://www.knjizare-vulkan.rs/ using scrapy.
The collected data has been stored in a relational database MySQL with over 20,000 current book records.

### Regression Implementation
Developed an application that uses the filtered book database to apply multiple linear regression. This models the relationship between various input features (author, genre, publisher, year of publication, number of pages, cover type, and format) and the book price. The model is trained using gradient descent and can predict book prices based on user-input features.


### Classification Implementation
Enhanced the application to include logistic regression for classifying book prices into different ranges. Implemented both "one-vs-all" and "multinomial logistic regression" approaches. The application allows users to choose the classification approach when inputting book attributes.

### Clustering Implementation
Applied the k-means clustering algorithm on at least three input features from the filtered book database. The clustering results provide insights into data distribution and segmentation.

## Repository Contents
* data_collection/: Contains the web crawler and scraper scripts along with the database schema and data import scripts.
* regression/: Code for the linear regression model, including data preprocessing, training, and prediction scripts.
* classification/: Implementation of logistic regression for book price classification.
* clustering/: Scripts for k-means clustering on book data.
