import os
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

class MentalHealthAssistant:
    def __init__(self, dataset_path='mental_health_dataset.csv'):
        self.df = self.initialize_dataset(dataset_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.initialize_models()
        self.client = InferenceClient(
            token=os.getenv("HF_TOKEN"),
            model="facebook/blenderbot-400M-distill"
        )

    def initialize_dataset(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file {path} not found")
        return pd.read_csv(path)

    def initialize_models(self):
        processed_texts = [text.lower().strip() for text in self.df['Context']]
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)

    def get_response(self, user_input):
        try:
            processed_input = user_input.lower().strip()
            input_vector = self.vectorizer.transform([processed_input])
            similarities = cosine_similarity(input_vector, self.tfidf_matrix)
            best_match_idx = np.argmax(similarities)
            
            # Priority 1: High confidence dataset match
            if similarities[0, best_match_idx] > 0.5:
                return self.df.iloc[best_match_idx]['Response']
            
            # Priority 2: API call
            try:
                response = self.client.text_generation(
                    prompt=user_input,
                    max_new_tokens=150,
                    temperature=0.7
                )
                return response.strip()
            
            except Exception as api_error:
                # Priority 3: Low confidence dataset match
                if similarities[0, best_match_idx] > 0.3:
                    return self.df.iloc[best_match_idx]['Response']
                raise api_error
                
        except Exception as e:
            # Priority 4: Generic responses
            return random.choice([
                "Let me think about that... Could you rephrase?",
                "I want to make sure I understand. Can you elaborate?",
                "Let's focus on what's most important right now."
            ])

mental_health_assistant = MentalHealthAssistant()