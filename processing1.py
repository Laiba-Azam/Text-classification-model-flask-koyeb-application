
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+')

    # Function to remove URLs from text
    def remove_urls(text):
        return url_pattern.sub('', text)

    # Remove URLs
    text = remove_urls(text)

    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numerical digits
    text = re.sub(r'\d', '', text)

    # Remove underscores
    text = text.replace('_', '')

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()

    # Function to convert NLTK's POS tags to WordNet's POS tags
    def nltk_to_wordnet_pos(nltk_tag):
        if nltk_tag.startswith('J'):
            return 'v'
        elif nltk_tag.startswith('V'):
            return 'v'
        elif nltk_tag.startswith('N'):
            return 'n'
        elif nltk_tag.startswith('R'):
            return 'r'
        else:
            return None

    # POS tagging and lemmatization
    nltk_tags = nltk.pos_tag(filtered_words)
    wordnet_tags = [(word, nltk_to_wordnet_pos(tag)) for word, tag in nltk_tags]
    lemmatized_words = [lemmatizer.lemmatize(word, pos=pos_tag) if pos_tag else word for word, pos_tag in wordnet_tags]

    # Return preprocessed text
    return ' '.join(lemmatized_words)


#with open('/content/drive/MyDrive/preprocess_text.pkl', 'wb') as file:
#    pickle.dump(preprocess_text, file)
