import streamlit as st

def login():
    # Hardcoded users for demonstration; replace with your user management system
    usernames = {"Derek": "coding_is_cool", "Kelsey": "hasaniceass"}

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
