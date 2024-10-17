import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def login():
    if 'auth_key' not in st.session_state:
        # Use AUTH_KEY from environment variables
        auth_key = os.getenv('AUTH_KEY')
        st.session_state.auth_key = auth_key

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    usernames = {"Derek": st.session_state.auth_key, 'CMSE830': st.session_state.auth_key}

    # Login form
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if username in usernames and usernames[username] == password:
            st.session_state.logged_in = True
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid credentials")
            return False

    return st.session_state.logged_in
