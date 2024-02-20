# Your existing imports
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from threading import Thread
import time

# Import your Chatbot and ChatbotTraining classes
from model_6 import Chatbot, ChatbotTraining

# Your existing code for creating the Flask app, SocketIO, and training thread
app = Flask(__name__)
socketio = SocketIO(app)

chatbot_training = None
chatbots = {}

def train_chatbot():
    global chatbot_training
    translation_model_name = "Helsinki-NLP/opus-mt-en-es"
    source_language_code = "es"
    target_language_code = "en"
    model_name = "anakin87/zephyr-7b-alpha-sharded"
    folder_path = '/content/drive/MyDrive/chatbot_data/'
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

    chatbot_training = ChatbotTraining(translation_model_name, source_language_code, target_language_code, model_name, folder_path, embedding_model_name)

# Thread for training the chatbot only once
train_thread = Thread(target=train_chatbot)
train_thread.start()

# Update the '/templates' route to render the 'index.html' file
@app.route('/templates')
def template():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    chatbots[user_id] = Chatbot(translation_model_name, source_language_code, target_language_code,
                                chatbot_training.model, chatbot_training.tokenizer,
                                chatbot_training.retriever, chatbot_training.llm, [])
    emit('connected', {'user_id': user_id})

@socketio.on('message')
def handle_message(data):
    user_id = request.sid
    if user_id in chatbots:
        query = data['query']
        use_spanish = data['useSpanish']
        chatbot_instance = chatbots[user_id]

        if use_spanish:
            translated_query = chatbot_instance.translate(query)
        else:
            translated_query = query

        response, chat_history = chatbot_instance.create_conversation(translated_query)

        if use_spanish:
            translated_response = chatbot_instance.translate(response)
        else:
            translated_response = response

        emit('response', {'response': translated_response})

# Your existing code for handling disconnect

if __name__ == '__main__':
    socketio.run(app, debug=True)
