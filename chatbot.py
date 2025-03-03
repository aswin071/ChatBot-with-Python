#Steps to build a chatbot using spaCy
#1)Prepare the Dataset
#2) Preprocess the Data and Train the Model
#3) Implement the Chatbot



import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nlp = spacy.load('en_core_web_sm')
# Sample dataset
data = [
    ("how can i reset my password", "reset_password"),
    ("i forgot my password", "reset_password"),
    ("reset my login", "reset_password"),
    ("what’s the status of my order", "order_status"),
    ("where is my order", "order_status"),
    ("track my package", "order_status"),
    ("how can i contact support", "contact_support"),
    (" ", "contact_support"),
    ("get me support", "contact_support"),
    ("are you open today", "store_hours"),
    ("what are your hours", "store_hours"),
    ("when do you close", "store_hours"),
    
]

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42) 
print('training data:', train_data)
train_texts, train_labels = zip(*train_data) 
test_texts, test_labels = zip(*test_data)  

# Create a text classification pipeline 
#A pipeline in scikit-learn is a way to chain multiple steps into one smooth process.
#The line  creates a single object (model) that handles both steps together.

model = make_pipeline(CountVectorizer(), MultinomialNB())

#Train the model
model.fit(train_texts,train_labels)

#Test the model
predicted_labels= model.predict(test_texts)
print("Test data:", predicted_labels)
for text, label, prediction in zip(test_texts, test_labels, predicted_labels):     
  print(f"Text: {text}\nActual: {label}\nPredicted: {prediction}\n")  

# Check model accuracy 
accuracy = accuracy_score(test_labels, predicted_labels) 
print(f"Model Accuracy: {accuracy * 100:.2f}%")




#Implement Chatbot
# Chatbot implementation
def get_intent(text):
    return model.predict([text])[0]
def get_response(intent):
    responses = {
        "reset_password": "To reset your password, click on 'Forgot Password' at the login screen and follow the instructions.",
        "order_status": "You can check the status of your order in the 'My Orders' section of our website.",
        "contact_support": "You can contact support via email at support@example.com or call us at 123-456-7890."
    }
    return responses.get(intent, "I'm not sure how to help with that. Please contact support directly at support@example.com.")

def chatbot():
    print("Welcome to Support Chatbot. How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        intent = get_intent(user_input)
        response = get_response(intent)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()