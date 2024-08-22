import streamlit as st
import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import httpx
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # intial config for error logs.

# Streamlit app configuration, it will host the Assistant Application.
st.set_page_config(page_title="Assistant", page_icon="🤖")
st.title("Assistant")

def get_response(user_query, chat_history): # this function return API response to streamlit and langchain for conversation.
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """
    # Serialize chat history
    serialized_chat_history = [{"role": "assistant" if isinstance(msg, AIMessage) else "user", "content": msg.content} for msg in chat_history]

    # Prepare request payload
    payload = {
        "model": "gpt-4",  # Use your model here
        "messages": serialized_chat_history + [{"role": "user", "content": user_query}],
        "temperature": 0.2
    }
    
    logging.debug("Payload for OpenAI API request: %s", payload) # IF issue persist in API request error will be loged.
    print("Hello Word", os.getenv("OPENAI_API_KEY"))
    # Make request to OpenAI API
    try:
        response = httpx.post(
            'https://api.openai.com/v1/chat/completions',
            json=payload,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            timeout=60.0
            verify=False  # Disable SSL verification
        )
        
        logging.debug("OpenAI API response: %s", response.json()) # it return response in JSON format for log analysis.
        return response.json()
    except Exception as e:
        logging.error("Error during OpenAI API request: %s", e)
        return {"error": str(e)} # it return error message with step where it is failing and what's the issue.

# Initialize session state
# below 'IF' check the start session is a new chat session.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation history
# if chat history is persist then it execute the for loop till session value.
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        if 'choices' in response and len(response['choices']) > 0:
            ai_response = response['choices'][0]['message']['content']
            st.write(ai_response)
            st.session_state.chat_history.append(AIMessage(content=ai_response))
        else:
            error_message = response.get("error", "Sorry, I couldn't generate a response. Please try again.")
            st.write(error_message)
            logging.error("No choices in the response or other error: %s", response)
