# SentiMart 📦
---
## Description ℹ️:
The Amazon Reviews Sentiment Analysis App is a user-friendly web application designed to analyze sentiment in Amazon product reviews. This is done by using advanced natural language processing (NLP) techniques, the app provides valuable insights into customer sentiments, helping users understand the reception of products on the Amazon platform. Whether you're a shopper seeking insights before making a purchase decision or a seller looking to gauge customer feedback, this app offers an intuitive interface for sentiment analysis.

## Demo Video

[![PRODUCT REVIEW](https://github.com/SaadARazzaq/SentiMart/assets/123338307/caae7a71-9c4c-46d0-9f33-1a6c1f6051e0)](https://www.linkedin.com/feed/update/urn:li:activity:7153314889896280066/)

## Libraries Used 📚:
- `requests`: For making HTTP requests to retrieve Amazon product review pages.
- `pandas`: For data manipulation and analysis.
- `BeautifulSoup`: For parsing HTML content.
- `datetime`: For handling date and time data.
- `re`: For regular expression operations.
- `nltk`: For natural language processing tasks such as tokenization, stopword removal, and lemmatization.
- `TextBlob`: For sentiment analysis.
- `matplotlib` and `seaborn`: For data visualization.
- `WordCloud`: For generating word clouds.
- `streamlit`: For building the interactive web application.

## Functionality ⚙️:
- **Import CSV**: Allows users to upload a CSV file containing Amazon reviews for sentiment analysis.
- **Write Review**: Enables users to input a review text manually for sentiment analysis.
- **Enter Amazon URL**: Provides users with the option to enter the URL of an Amazon product page to scrape reviews and perform sentiment analysis.
- **Sentiment Analysis**: Analyzes the sentiment of reviews and presents the result along with confidence scores.
- **Visualizations**: Offers various visualizations including sentiment distribution, pie chart, histogram, review length distribution, comparison of sentiment across products, time series analysis of sentiment, and keyword frequency analysis.

## Usage 🖥️:
1. Choose an option from the sidebar: You can Import CSV, Write Review, or Enter Amazon URL.
2. Based on the selected option, upload a CSV file, enter a review, or input an Amazon URL.
3. Click the "Analyze" button to perform sentiment analysis.
4. Explore the sentiment analysis result and visualizations to gain insights into customer sentiments.

## Technologies Used 💻:
- Python
- Streamlit: For creating interactive web applications in Python.
- BeautifulSoup: For web scraping and parsing HTML content.
- NLTK (Natural Language Toolkit): For natural language processing tasks.
- TextBlob: For sentiment analysis and text processing.
- Matplotlib and Seaborn: For data visualization.
- WordCloud: For generating word clouds.

## Note 📌:
Ensure that all required libraries are installed in your Python environment before running the application.
