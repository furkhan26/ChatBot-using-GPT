from flask import Flask, render_template, request
from MyChat import ChatBot

app = Flask(__name__)
app.static_folder = 'static' 
chatbot = ChatBot() 

@app.route("/")
def index():                   
    return render_template('chat.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["userInput"]
    response = chatbot.generate_response(user_input)
    print(response)
    return {"response": response}

    
if __name__ == "__main__":
    app.run(debug=True)