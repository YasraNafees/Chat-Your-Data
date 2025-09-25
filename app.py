import streamlit as st
from backend import setup_pipeline_and_query, PDF_PATH

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Load previous messages
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        response = st.write_stream(
            chunk.content for chunk in setup_pipeline_and_query(
                pdf_path=PDF_PATH,
                question=user_input
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': response})
