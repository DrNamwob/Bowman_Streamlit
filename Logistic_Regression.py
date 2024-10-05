import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd


dictionary = pd.read_csv('/Users/derekebowman/Coding Projects/Bowman_Streamlit/dictionary.csv')

# Define a color map
colors = {'M': 'blue', 'F': 'pink'}  # Customize colors as needed

# Get the color for each point based on sex
point_colors = [colors[sex] for sex in dictionary['Sex']]

male_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Male')
female_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Female')


fig, ax = plt.subplots()

plt.scatter(dictionary['Diet'], dictionary['MAP'], color =point_colors)
plt.legend(['Blue'],['Pink'])
plt.title('MAP BP at 24W')

plt.legend(handles=[male_handle, female_handle])

# Title and labels
plt.title('MAP BP at 24W')
plt.xlabel('Diet')
plt.ylabel('MAP')

# Show plot
plt.xticks(rotation=45)  # Rotate x-tick labels if needed
plt.tight_layout()  # Adjust layout for better visibility
plt.show()


st.pyplot(fig)

