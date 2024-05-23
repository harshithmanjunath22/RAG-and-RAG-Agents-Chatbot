import streamlit as st
import random
from streamlit_chat import message
from ragagents import get_response

@st.cache_data
def cached_get_response(user_input):
    response = get_response({"input": user_input})
    return response.get("output")


st.set_page_config(page_title="💬Chatbot")
st.header("hsag Energy Industry Chatbot")
#st.markdown("Ask your Energy related questions here.")

with st.sidebar:
    st.title('hsag💬Chatbot')
    st.image('https://hsag.info/wp-content/uploads/2020/09/hsag_bildmarke_sz_4c.png', caption='Heidelberger Services AG')
    st.write('This chatbot is created using the Azure OPENAI GPT 3.5Turbo LLM model.')

    

if "messages" not in st.session_state.keys():
    st.session_state.messages = [ 
        {"role": "assistant", "content": "Hi, I am your chatbot, ask me your queries. I am here to assist you. My knowledge related to energy industry is vast!"}
    ]
    
    st.session_state.chat_engine = get_response
    

    st.session_state.chat_engine = get_response
    
if prompt := st.chat_input("Enter your Prompt here"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Incorporating the conversation memory snippet
            user_input = st.session_state.messages[-1]["content"]
            response = cached_get_response(prompt)
            #response = get_response(user_input)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            