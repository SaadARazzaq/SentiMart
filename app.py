import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def get_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def get_reviews_url():
    return 'https://www.amazon.com/Fitbit-Smartwatch-Readiness-Exercise-Tracking/product-reviews/B0B4MWCFV4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

def reviewsHtml(url, len_page):
    headers = get_headers()
    soups = []
    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        soups.append(soup)
    return soups

def get_reviews_data(html_data):
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')
    for box in boxes:
        try:
            name = box.select_one('[class="a-profile-name"]').text.strip()
        except Exception as e:
            name = 'N/A'
        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
        except Exception as e:
            stars = 'N/A'   
        try:
            title = box.select_one('[data-hook="review-title"]').text.strip()
        except Exception as e:
            title = 'N/A'
        try:
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'
        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'
        data_dict = {
            'Name' : name,
            'Stars' : stars,
            'Title' : title,
            'Date' : date,
            'Description' : description
        }
        data_dicts.append(data_dict)
    return data_dicts

def process_data(html_datas, len_page):
    reviews = []
    for html_data in html_datas:
        review = get_reviews_data(html_data)
        reviews += review
    df_reviews = pd.DataFrame(reviews)
    return df_reviews

def clean_data(df_reviews):
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: x.lower())
    stop_words = set(stopwords.words('english'))
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
    df_reviews.to_csv('cleaned_reviews.csv', index=False)
    print("Data processing and cleaning completed.")
    return df_reviews

def analyze_sentiment(description):
    analysis = TextBlob(description)
    sentiment = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    confidence = abs(sentiment) + (1 - subjectivity) * 100
    
    if sentiment > 0:
        return 'Positive', confidence
    elif sentiment < 0:
        return 'Negative', confidence
    else:
        return 'Neutral', confidence

def train_data(df_reviews):
    df_reviews[['Sentiment', 'Confidence']] = df_reviews['Description'].apply(analyze_sentiment).apply(pd.Series)
    return df_reviews[['Description', 'Sentiment', 'Confidence']]

def visualize_data(df_reviews):
    st.subheader("Visualized Data:")

    st.subheader("Sentiment Distribution:")
    info_text = '''
                - This visualization represents the distribution of sentiment categories in the reviews.
                - Each bar represents a different sentiment category: Positive, Negative, or Neutral.
                - The size of each bar indicates the proportion of reviews belonging to that sentiment category.
                - For example, if the "Positive" bar is larger, it means there are more positive reviews compared to negative or neutral ones
    '''
    with st.expander("💡Info"):
        st.write(info_text)

    sentiment_counts = df_reviews['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Pie Chart:")
    visualize_pie_chart(df_reviews)

    st.subheader("Histogram:")
    visualize_histogram(df_reviews)

    st.subheader("Distribution of Review Length:")
    visualize_review_length_distribution(df_reviews)
            
    st.subheader("Comparison of Sentiment Across Products:")
    compare_sentiment_across_products(df_reviews)

    st.subheader("Time Series Analysis of Product:")
    visualize_time_series(df_reviews)

    st.subheader("Keyword Frequency Analysis:")
    all_words = ' '.join(df_reviews['Description'])
    generate_wordcloud_st(all_words)

def visualize_pie_chart(df_reviews):
    info_text = '''
        - This chart is like a pizza divided into slices.
        - Each slice represents a different sentiment category: Positive, Negative, or Neutral.
        - The size of each slice shows how many reviews fall into that sentiment category.
'''
    with st.expander("💡Info"):
        st.write(info_text)

    sentiment_counts = df_reviews['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'), startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

def visualize_histogram(df_reviews):
    info_text = '''
        - Imagine stacking blocks to make a bar graph.
        - Each block represents the number of reviews with a specific confidence score.
        - The height of each bar tells us how many reviews have a certain level of confidence in their sentiment analysis.
        - For example, if a bar is tall, it means many reviews have high confidence in their sentiment analysis, while a shorter bar means fewer reviews have high confidence.
        - This helps us understand the distribution of confidence scores among the reviews.
'''
    with st.expander("💡Info"):
        st.write(info_text)


    plt.figure(figsize=(10, 6))
    sns.histplot(df_reviews['Confidence'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Sentiment Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    st.pyplot()

def analyze_sentiment_st(description):
    analysis = TextBlob(description)
    sentiment = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    confidence = abs(sentiment) + (1 - subjectivity) * 100
    
    if sentiment > 0:
        return 'Positive', confidence
    elif sentiment < 0:
        return 'Negative', confidence
    else:
        return 'Neutral', confidence

def generate_wordcloud_st(words):
    info_text = '''
        - This shows us which words appear most often in the reviews.
        - Think of it as finding the most popular words in a book.
        - The bigger the word in the cloud, the more often it appears in the reviews.
'''
    with st.expander("💡Info"):
        st.write(info_text)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

def visualize_time_series(df):
    info_text = '''
        - Think of this visualization as a tool to see how sentiments (like positivity, neutrality, or negativity) change over time.
        - Imagine a graph with lines showing how people's feelings about the product evolve from day to day.
        - Each line on the graph represents a type of sentiment: positive, neutral, or negative.
        - The horizontal line represents dates, so you can see how sentiments change over different days.
        - The vertical line shows the number of reviews, giving an idea of how many people feel a certain way each day.
        - This graph helps us understand if people's feelings about something are changing over time.
'''
    with st.expander("💡Info"):
        st.write(info_text)
        
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")

    # df['Date'] = pd.to_datetime(df['Date'])
    df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=['Negative', 'Neutral', 'Positive'], ordered=True)
    df_time_series = df.groupby([pd.Grouper(key='Date', freq='D'), 'Sentiment']).size().unstack(fill_value=0)
    df_time_series.plot(kind='line', stacked=True, figsize=(10, 6))
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    st.pyplot()

def visualize_review_length_distribution(df):
    info_text = '''
        - Think of this visualization as a way to understand the distribution of review lengths.
        - Review length refers to the number of words in each review.
        - Frequency in this context means how often reviews of different lengths occur.
        - Imagine a line graph where the length of the line at each point represents the frequency of reviews with a specific length.
        - Longer parts of the line mean more reviews are that length, while shorter parts mean fewer reviews are that length.
        - For example, if you see a tall peak in the graph, it means many reviews are of that length, while a flat area indicates fewer reviews of that length.
        - This helps us understand how long or short the reviews are on average and how common reviews of different lengths are.
'''
    with st.expander("💡Info"):
        st.write(info_text)

    df['Review Length'] = df['Description'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Review Length'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Review Length')
    plt.xlabel('Review Length')
    plt.ylabel('Frequency')
    st.pyplot()

def compare_sentiment_across_products(df):
    info_text = '''
        - This visualization compares the sentiment of reviews for different products.
        - Imagine comparing how people feel about various items or services.
        - Each bar on the chart represents the number of positive, negative, and neutral reviews for each product.
        - For example, if you see a tall blue section (positive sentiment) on a bar, it means many reviews for that product are positive.
        - This comparison helps us understand the overall sentiment distribution across different products.
'''
    with st.expander("💡Info"):
        st.write(info_text)


    sentiment_counts_by_product = df.groupby('Name')['Sentiment'].value_counts().unstack(fill_value=0)
    sentiment_counts_by_product.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Sentiment Comparison Across Products')
    plt.xlabel('Product')
    plt.ylabel('Number of Reviews')
    st.pyplot()

def visualize_keyword_frequency(df):
    info_text = '''
        - This shows us which words appear most often in the reviews.
        - Think of it as finding the most popular words in a book.
        - The bigger the word in the cloud, the more often it appears in the reviews.
'''
    with st.expander("💡Info"):
        st.write(info_text)

    all_words = ' '.join(df['Description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

def import_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_and_store_data(df, csv_filename='cleaned_reviews.csv'):
    # Clean data
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df['Description'] = df['Description'].apply(lambda x: x.lower())
    stop_words = set(stopwords.words('english'))
    df['Description'] = df['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df['Description'] = df['Description'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    # Store cleaned data in a new CSV
    cleaned_csv_path = csv_filename
    df.to_csv(cleaned_csv_path, index=False)
    
    return cleaned_csv_path

def main():

    st.title("SentiMart📦: Amazon Sentiment App")

    option = st.sidebar.selectbox("Choose an option", ["Write Review", "Enter Amazon URL", "Import CSV"])

    if option == "Import CSV":
        st.header("Import CSV for Analysis")

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            df[['Sentiment', 'Confidence']] = df['Description'].apply(analyze_sentiment_st).apply(pd.Series)

            st.subheader("Data Preview:")
            st.write(df.head())

            st.subheader("Visualized Data:")
            
            st.subheader("Sentiment Distribution:")
            info_text = '''
                - This visualization represents the distribution of sentiment categories in the reviews.
                - Each bar represents a different sentiment category: Positive, Negative, or Neutral.
                - The size of each bar indicates the proportion of reviews belonging to that sentiment category.
                - For example, if the "Positive" bar is larger, it means there are more positive reviews compared to negative or neutral ones
            '''
            with st.expander("💡Info"):
                st.write(info_text)

            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts)

            st.subheader("Pie Chart:")
            visualize_pie_chart(df)

            st.subheader("Histogram:")
            visualize_histogram(df)
            
            st.subheader("Distribution of Review Length:")
            visualize_review_length_distribution(df)
            
            st.subheader("Comparison of Sentiment Across Products:")
            compare_sentiment_across_products(df)

            st.subheader("Time Series Analysis of Product:")
            visualize_time_series(df)
            
            st.subheader("Keyword Frequency Analysis:")
            visualize_keyword_frequency(df)

    elif option == "Write Review":
        st.header("Write Review for Analysis")

        user_input = st.text_area("Enter your review:")

        if st.button("Analyze"):
            if user_input:
                result, confidence = analyze_sentiment_st(user_input)
                st.subheader("Sentiment Analysis Result:")
                st.write(f"Sentiment: {result}")
                st.write(f"Confidence Score: {confidence}")

            else:
                st.warning("Please enter a review for analysis.")

    elif option == "Enter Amazon URL":
        st.header("Enter Your Favourite Amazon product's URL")

        try:
            URL_input = st.text_input("Enter Valid Amazon URL:")
        except ValueError as e:
            st.warning("Error: "+e)

        page_len = st.slider("Select the number of pages to scrape", min_value=1, max_value=10, value=1)

        if st.button("Analyze"):
            if URL_input:
                html_datas = reviewsHtml(URL_input, page_len)
                df_reviews = process_data(html_datas, page_len)
                df_reviews = clean_data(df_reviews)
                cleaned_csv_path = clean_and_store_data(df_reviews)

                df_cleaned = import_data(cleaned_csv_path)
                df_cleaned[['Sentiment', 'Confidence']] = df_cleaned['Description'].apply(analyze_sentiment_st).apply(pd.Series)

                st.subheader("Data Preview after Cleaning:")

                st.write(df_cleaned.head())

                visualize_data(df_cleaned)

            else:
                st.warning("Please enter a URL first!")

if __name__ == "__main__":
    main()
