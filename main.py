import json
import os 
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
Answer the question below:
Here is the conversation history: {contesxt}
Question: {question}
Answer:
"""

model = OllamaLLM(model = "llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def load_conversation_history(filename="conversation_history.json"):
    if os.path.exists(filename):
        with open(filename,"r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return{}       ##return empty / corrupted file

    else : ##creating the file with empty json if file not exist
        with open(filename, "w") as file:  
            json.dump({}, file)
            return{}            


def save_conversation_history(history, filename="conversation_history.json"):
    with open(filename, "w") as file:
        json.dump(history, file, indent=4)


def handle_conversation():
    conversation_history = load_conversation_history()
    context = ""
    if conversation_history:
        for user_input, result in conversation_history.items():
            context += f"\nUser: {user_input}\nAI: {result}"

    print("Welcome to the llama3 based chatbot....Type 'exit' to quit. ")
    while True:
        print("Waiting for user input...")
        user_input = input("You: ")
        if user_input.lower() == "exit": ##exit funtion
            break 

        if user_input in conversation_history: ##handling repated questions
            result = conversation_history[user_input]
            print("Bot: ", result)
        else: ##new question
            result = chain.invoke({"contesxt": context, "question": user_input})
            print("Bot: ", result)
            conversation_history[user_input] = result ##save the new converstion to history

        context += f"\nUser: {user_input}\nAI: {result}"

        save_conversation_history(conversation_history) ##saving convo when chat ends


if __name__ =="__main__":
    handle_conversation()


