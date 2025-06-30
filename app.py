from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ml.sentiment import analyze_sentiment, detect_toxicity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    user = data['user']
    msg = data['message']

    sentiment = analyze_sentiment(msg)
    toxicity, score = detect_toxicity(msg)

    response = {
        'user': user,
        'message': msg,
        'sentiment': sentiment,
        'toxicity': toxicity,
        'score': score
    }
    emit('response', response, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)
