import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def login():
    # Use AUTH_KEY from environment variables
    auth_key = os.getenv('AUTH_KEY')
    usernames = {"Derek": auth_key, 'CMSE830': "GoGreen1"}

    # Login form
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if username in usernames and usernames[username] == password:
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid credentials")
            return False
