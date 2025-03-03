# import flask
from flask import Flask, request, jsonify
from chatbot import *
# initialize the app

app = Flask(__name__)
# create api endpint

@app.route('/chatbot', methods=['POST'])
def chatbotapi():
    data = request.get_json()
    user_input = data['user_input']
    intent = get_intent(user_input)
    response = get_response(intent)
    return jsonify({'response': response})


if __name__ == "__main__":
    app.run(debug=True)