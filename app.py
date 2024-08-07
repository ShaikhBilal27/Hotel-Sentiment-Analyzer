from flask import Flask, render_template, request, jsonify
import csv
from collections import Counter
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLP tools
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize ML models
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()
regressor = LinearRegression()

def get_sentiment_analysis(review):
    return sia.polarity_scores(review)

def calculate_sentiment_percentages(sentiments):
    total = len(sentiments)
    sentiment_counts = Counter(sentiments)
    positive_percent = (sentiment_counts['positive'] / total) * 100
    negative_percent = (sentiment_counts['negative'] / total) * 100
    neutral_percent = (sentiment_counts['neutral'] / total) * 100
    return positive_percent, negative_percent, neutral_percent

def read_csv_file(filename):
    data = {}
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            hotel = row['Hotel']
            dish = row['Dish']
            review = row['Review']
            if hotel not in data:
                data[hotel] = {}
            if dish not in data[hotel]:
                data[hotel][dish] = []
            data[hotel][dish].append(review)
    return data

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def simple_summarize(text, num_sentences=2):
    sentences = text.split('.')
    return '. '.join(sentences[:num_sentences]) + '.'

# Load data
hotel_data = read_csv_file('hotels_review.csv')

# Prepare data for ML models
X = [review for hotel in hotel_data.values() for dish in hotel.values() for review in dish]
y_sentiment = [1 if get_sentiment_analysis(review)['compound'] > 0 else 0 for review in X]  # Binary sentiment
y_rating = [random.randint(1, 5) for _ in X]  # Simulated ratings, replace with actual ratings if available

# Train ML models
X_train, X_test, y_sentiment_train, y_sentiment_test, y_rating_train, y_rating_test = train_test_split(
    X, y_sentiment, y_rating, test_size=0.2, random_state=42)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

classifier.fit(X_train_vectorized, y_sentiment_train)
regressor.fit(X_train_vectorized, y_rating_train)

@app.route('/')
def index():
    hotels = list(hotel_data.keys())
    return render_template('index.html', hotels=hotels)

@app.route('/get_menu', methods=['POST'])
def get_menu():
    hotel = request.form['hotel']
    menu = list(hotel_data[hotel].keys())
    return jsonify(menu)

@app.route('/analyze_dish', methods=['POST'])
def analyze_dish():
    hotel = request.form['hotel']
    dish = request.form['dish']
    reviews = hotel_data[hotel][dish]
    
    sentiments = [get_sentiment_analysis(review)['compound'] for review in reviews]
    positive = [review for review, sentiment in zip(reviews, sentiments) if sentiment > 0.05]
    negative = [review for review, sentiment in zip(reviews, sentiments) if sentiment < -0.05]
    neutral = [review for review, sentiment in zip(reviews, sentiments) if -0.05 <= sentiment <= 0.05]
    
    positive_percent, negative_percent, neutral_percent = calculate_sentiment_percentages(['positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral' for s in sentiments])
    
    # Use ML models for additional insights
    vectorized_reviews = vectorizer.transform(reviews)
    predicted_sentiments = classifier.predict(vectorized_reviews)
    predicted_ratings = regressor.predict(vectorized_reviews)
    
    # Generate simple summary
    summary = simple_summarize(' '.join(reviews))
    
    return jsonify({
        'positive': random.choice(positive) if positive else "No positive review",
        'negative': random.choice(negative) if negative else "No negative review",
        'neutral': random.choice(neutral) if neutral else "No neutral review",
        'score': sum(sentiments) / len(sentiments),
        'positive_percent': positive_percent,
        'negative_percent': negative_percent,
        'neutral_percent': neutral_percent,
        'predicted_sentiment': np.mean(predicted_sentiments),
        'predicted_rating': np.mean(predicted_ratings),
        'summary': summary
    })

@app.route('/recommend_food', methods=['POST'])
def recommend_food():
    hotel = request.form['hotel']
    menu = hotel_data[hotel]
    recommendations = []

    for dish, reviews in menu.items():
        sentiments = [get_sentiment_analysis(review)['compound'] for review in reviews]
        score = sum(sentiments) / len(sentiments)
        positive_percent, negative_percent, neutral_percent = calculate_sentiment_percentages(['positive' if s > 0.05 else 'negative' if s < -0.05 else 'neutral' for s in sentiments])
        
        # Use ML models for predictions
        vectorized_reviews = vectorizer.transform(reviews)
        predicted_sentiments = classifier.predict(vectorized_reviews)
        predicted_ratings = regressor.predict(vectorized_reviews)
        
        recommendations.append({
            'dish': dish,
            'score': score,
            'positive_percent': positive_percent,
            'negative_percent': negative_percent,
            'neutral_percent': neutral_percent,
            'predicted_sentiment': np.mean(predicted_sentiments),
            'predicted_rating': np.mean(predicted_ratings)
        })

    recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return jsonify(recommendations[:3])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    processed_input = preprocess_text(user_input)
    
    # Intent classification
    if any(word in processed_input for word in ['hello', 'hi', 'hey']):
        return jsonify({'response': "Hello! How can I assist you with your food choices today?"})
    
    elif 'best' in processed_input:
        for dish in set(dish for hotel in hotel_data.values() for dish in hotel.keys()):
            if dish.lower() in processed_input:
                best_hotel, best_reviews = max(((hotel, reviews) for hotel, menu in hotel_data.items() for d, reviews in menu.items() if d.lower() == dish.lower()), key=lambda x: np.mean(regressor.predict(vectorizer.transform(x[1]))))
                best_score = np.mean(regressor.predict(vectorizer.transform(best_reviews)))
                response = f"The best {dish} can be found at {best_hotel} with a predicted rating of {best_score:.2f}/5. "
                summary = simple_summarize(' '.join(best_reviews))
                response += f"Here's a summary of the reviews: {summary}"
                return jsonify({'response': response})
        return jsonify({'response': "I couldn't identify a specific dish in your question. Could you please specify which dish you're looking for?"})
    
    elif 'recommend' in processed_input:
        for hotel in hotel_data.keys():
            if hotel.lower() in processed_input:
                recommendations = recommend_food()
                response = f"Here are the top 3 recommended dishes at {hotel}:\n"
                for rec in recommendations[:3]:
                    response += f"- {rec['dish']} (Predicted rating: {rec['predicted_rating']:.2f}/5)\n"
                return jsonify({'response': response})
        return jsonify({'response': "I'd be happy to recommend some dishes! Please specify a hotel, and I'll provide you with our top recommendations based on customer reviews."})
    
    elif 'review' in processed_input or 'sentiment' in processed_input or 'score' in processed_input:
        for hotel in hotel_data.keys():
            if hotel.lower() in processed_input:
                for dish in hotel_data[hotel].keys():
                    if dish.lower() in processed_input:
                        analysis = analyze_dish()
                        response = f"Analysis for {dish} at {hotel}:\n"
                        response += f"Overall sentiment score: {analysis['score']:.2f}\n"
                        response += f"Predicted rating: {analysis['predicted_rating']:.2f}/5\n"
                        response += f"Positive: {analysis['positive_percent']:.1f}%, Negative: {analysis['negative_percent']:.1f}%, Neutral: {analysis['neutral_percent']:.1f}%\n"
                        response += f"Summary: {analysis['summary']}\n"
                        response += f"Sample positive review: {analysis['positive']}\n"
                        response += f"Sample negative review: {analysis['negative']}"
                        return jsonify({'response': response})
                return jsonify({'response': f"I can provide analysis for dishes at {hotel}. Which specific dish would you like to know about?"})
        return jsonify({'response': "I can provide detailed analysis of reviews and sentiment scores for specific dishes or hotels. Which one would you like to know about?"})
    
    elif 'thank' in processed_input:
        return jsonify({'response': "You're welcome! Enjoy your meal and don't hesitate to ask if you need anything else."})
    
    else:
        return jsonify({'response': "I'm sorry, but I'm currently unable to answer general questions. Please ask about specific hotels, dishes, or recommendations."})

if __name__ == '__main__':
    app.run(debug=True)