import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import random

# Load the dataset
file_path = 'dataset.csv'
sample_size = 5000  # Adjust the sample size as needed

# Set a random seed for reproducibility
random.seed(42)

# Read a random sample of rows from the dataset, skipping problematic rows
df = pd.read_csv(
    file_path,
    skiprows=lambda i: i > 0 and random.random() > (sample_size / 30000),
    on_bad_lines='skip'  # Skips rows with unexpected number of columns
)

# Fill missing values for specific columns
df['Hotel Details'].fillna('Not Available', inplace=True)
df['Airline'].fillna('Not Available', inplace=True)
df['Onwards Return Flight Time'].fillna('Not Available', inplace=True)
df['Sightseeing Places Covered'].fillna('Not Available', inplace=True)
df['Initial Payment For Booking'].fillna(0, inplace=True)
df['Cancellation Rules'].fillna('Not Available', inplace=True)

# Drop columns with all missing values
df.drop(columns=["Flight Stops", "Meals", "Initial Payment For Booking", "Date Change Rules", "Unnamed: 22", "Unnamed: 23"], inplace=True, errors='ignore')

# Convert 'Travel Date' column to datetime
df['Travel Date'] = pd.to_datetime(df['Travel Date'], format='%d-%m-%Y', errors='coerce')

# Filter for allowed package types
allowed_package_types = ['Deluxe', 'Standard', 'Premium', 'Luxury', 'Budget']
df = df[df['Package Type'].isin(allowed_package_types)]

# Drop unnecessary columns
df.drop(columns=['Company', 'Crawl Timestamp'], inplace=True, errors='ignore')

# Preprocess data for hotel recommendations
df['Hotel_Info'] = df['Hotel Details'].str.cat(df['Destination'], sep='|')

# Create a TF-IDF vectorizer to convert text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the vectorizer on the Hotel_Info column
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Hotel_Info'])

# Compute the cosine similarity between hotels based on TF-IDF vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get hotel recommendations based on criteria
def get_hotel_recommendations(package_type, start_city, price, destination, cosine_sim=cosine_sim):
    # Filter the dataset based on the given criteria
    filtered_data = df[(df['Package Type'] == package_type) &
                       (df['Start City'] == start_city) &
                       (df['Price Per Two Persons'] <= price) &
                       (df['Destination'] == destination)]

    if filtered_data.empty:
        return "No matching hotels found."

    # Get the indices of the filtered hotels
    hotel_indices = filtered_data.index

    # Calculate the average cosine similarity score for each hotel with the filtered hotels
    avg_similarity_scores = []
    for idx in hotel_indices:
        avg_score = sum(cosine_sim[idx]) / len(cosine_sim[idx])
        avg_similarity_scores.append(avg_score)

    # Create a DataFrame to store the filtered hotels and their average similarity scores
    recommended_hotels_df = pd.DataFrame({
        'Uniq Id': filtered_data['Uniq Id'],
        'Hotel Details': filtered_data['Hotel Details'],
        'Avg Similarity Score': avg_similarity_scores
    })

    # Sort hotels by average similarity score in descending order
    recommended_hotels_df = recommended_hotels_df.sort_values(by='Avg Similarity Score', ascending=False)

    # Return the recommended hotel details
    return recommended_hotels_df[['Uniq Id', 'Hotel Details']]

# Streamlit application
st.title('Hotel Recommendation App')

# Dropdowns for user selections
package_types = df['Package Type'].unique()
start_cities = df['Start City'].unique()
destinations = df['Destination'].unique()

package_type = st.selectbox('Select Package Type:', package_types)
start_city = st.selectbox('Select Start City:', start_cities)
destination = st.selectbox('Select Destination:', destinations, format_func=lambda x: x.replace('|', ', '))

# Slider for Price
price = st.slider('Select Maximum Price:', min_value=0, max_value=int(df['Price Per Two Persons'].max()), value=10000)

# Button to get recommendations
if st.button('Get Recommendations'):
    recommended_hotels = get_hotel_recommendations(package_type, start_city, price, destination)
    if isinstance(recommended_hotels, str):
        st.warning(recommended_hotels)
    else:
        st.table(recommended_hotels)

# Display selected filters for user
st.write('Selected Filters:')
st.write(f'Package Type: {package_type}')
st.write(f'Start City: {start_city}')
st.write(f'Destination: {destination}')
st.write(f'Maximum Price: {price}')
