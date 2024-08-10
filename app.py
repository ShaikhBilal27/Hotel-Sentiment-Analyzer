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

def extract_attributes(reviews):
    positive_attributes = Counter()
    negative_attributes = Counter()
    for review in reviews:
        sentiment = get_sentiment_analysis(review)
        words = preprocess_text(review).split()
        for word in words:
            if sentiment['compound'] > 0.05:
                positive_attributes[word] += 1
            elif sentiment['compound'] < -0.05:
                negative_attributes[word] += 1
    return positive_attributes, negative_attributes

def generate_customer_summary(reviews):
    positive_attrs, negative_attrs = extract_attributes(reviews)
    summary = "Customers like the "
    summary += ", ".join(attr for attr, count in positive_attrs.most_common(5))
    summary += ". However, some customers mentioned issues with "
    summary += ", ".join(attr for attr, count in negative_attrs.most_common(3))
    summary += "."
    return summary

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
    
    total_reviews = len(reviews)
    
    # Use ML models for predictions
    vectorized_reviews = vectorizer.transform(reviews)
    predicted_sentiments = classifier.predict(vectorized_reviews)
    predicted_ratings = regressor.predict(vectorized_reviews)
    
    average_rating = np.mean(predicted_ratings)
    
    positive_attributes, negative_attributes = extract_attributes(reviews)
    
    customer_summary = generate_customer_summary(reviews)
    
    attributes_to_learn = ['Quality', 'Ease of use', 'Appearance', 'Value', 'Light', 'Safety']
    
    return jsonify({
        'total_reviews': total_reviews,
        'average_rating': average_rating,
        'customer_summary': customer_summary,
        'positive_attributes': dict(positive_attributes.most_common(5)),
        'negative_attributes': dict(negative_attributes.most_common(5)),
        'attributes_to_learn': attributes_to_learn
    })

@app.route('/recommend_food', methods=['POST'])
def recommend_food():
    hotel = request.form['hotel']
    menu = hotel_data[hotel]
    recommendations = []

    for dish, reviews in menu.items():
        vectorized_reviews = vectorizer.transform(reviews)
        predicted_ratings = regressor.predict(vectorized_reviews)
        average_rating = np.mean(predicted_ratings)
        
        positive_attributes, negative_attributes = extract_attributes(reviews)
        
        recommendations.append({
            'dish': dish,
            'average_rating': average_rating,
            'total_reviews': len(reviews),
            'top_positive': dict(positive_attributes.most_common(3)),
            'top_negative': dict(negative_attributes.most_common(3))
        })

    recommendations.sort(key=lambda x: x['average_rating'], reverse=True)
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
                customer_summary = generate_customer_summary(best_reviews)
                response += f"Here's what customers say: {customer_summary}"
                return jsonify({'response': response})
        return jsonify({'response': "I couldn't identify a specific dish in your question. Could you please specify which dish you're looking for?"})
    
    elif 'recommend' in processed_input:
        for hotel in hotel_data.keys():
            if hotel.lower() in processed_input:
                recommendations = recommend_food()
                response = f"Here are the top 3 recommended dishes at {hotel}:\n"
                for rec in recommendations[:3]:
                    response += f"- {rec['dish']} (Avg. rating: {rec['average_rating']:.2f}/5, {rec['total_reviews']} reviews)\n"
                    response += f"  Top positives: {', '.join(rec['top_positive'].keys())}\n"
                    response += f"  Top negatives: {', '.join(rec['top_negative'].keys())}\n"
                return jsonify({'response': response})
        return jsonify({'response': "I'd be happy to recommend some dishes! Please specify a hotel, and I'll provide you with our top recommendations based on customer reviews."})
    
    elif 'review' in processed_input or 'rating' in processed_input:
        for hotel in hotel_data.keys():
            if hotel.lower() in processed_input:
                for dish in hotel_data[hotel].keys():
                    if dish.lower() in processed_input:
                        analysis = analyze_dish()
                        response = f"Analysis for {dish} at {hotel}:\n"
                        response += f"Average rating: {analysis['average_rating']:.2f}/5 ({analysis['total_reviews']} reviews)\n"
                        response += f"Customer summary: {analysis['customer_summary']}\n"
                        response += "Top positive attributes: " + ", ".join(analysis['positive_attributes'].keys()) + "\n"
                        response += "Top negative attributes: " + ", ".join(analysis['negative_attributes'].keys())
                        return jsonify({'response': response})
                return jsonify({'response': f"I can provide analysis for dishes at {hotel}. Which specific dish would you like to know about?"})
        return jsonify({'response': "I can provide detailed analysis of reviews and ratings for specific dishes or hotels. Which one would you like to know about?"})
    
    elif 'thank' in processed_input:
        return jsonify({'response': "You're welcome! Enjoy your meal and don't hesitate to ask if you need anything else."})
    
    else:
        return jsonify({'response': "I'm sorry, but I'm currently unable to answer general questions. Please ask about specific hotels, dishes, or recommendations."})

if __name__ == '__main__':
    app.run(debug=True)