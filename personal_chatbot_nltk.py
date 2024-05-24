import nltk
from nltk.chat.util import Chat, reflections

# Define pairs of patterns and responses for the chatbot
pairs = [
    [
        r"hi|hello|hey",
        ["Hello!", "Hey there!", "Hi!"]
    ],
    [
        r"how are you ?",
        ["I'm doing well, thank you!", "I'm good, thanks for asking.", "All good, thanks!"]
    ],
    [
        r"what is your name ?",
        ["I'm a chatbot.", "You can call me Chatbot.", "I don't have a name, I'm just a bot."]
    ],
    [
        r"bye|goodbye",
        ["Goodbye!", "See you later!", "Bye!"]
    ],
    [
        r"(.*)",
        ["Sorry, I didn't understand that.", "Could you please rephrase that?", "I'm not sure I follow."]
    ]
]

# Create a chatbot using the Chat class from NLTK
chatbot = Chat(pairs, reflections)

def chatbot_response(user_input):
    return chatbot.respond(user_input)

# Test the chatbot
print("Chatbot: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    response = chatbot_response(user_input)
    print("Chatbot:", response)
    if user_input.lower() == "bye":
        break
