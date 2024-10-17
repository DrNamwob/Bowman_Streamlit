import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd

# Sample data for the visualizations
data_line = np.linspace(0, 10, 100)
data_bar = {'Fruits': ['Apples', 'Bananas', 'Cherries', 'Dates'], 'Counts': [10, 15, 7, 12]}
data_scatter = {'X Values': [1, 2, 3, 4, 5], 'Y Values': [2, 3, 5, 7, 11]}

# Create the tabs
tabs = st.tabs(["Line Chart", "Bar Chart", "Scatter Plot"])

# Tab 1: Line Chart
with tabs[0]:
    st.header("Line Chart Example")
    y_line = np.sin(data_line)
    fig_line = px.line(x=data_line, y=y_line, title='Line Chart of Sine Wave')
    st.plotly_chart(fig_line)

# Tab 2: Bar Chart
with tabs[1]:
    st.header("Bar Chart Example")
    df_bar = pd.DataFrame(data_bar)
    fig_bar = px.bar(df_bar, x='Fruits', y='Counts', title='Bar Chart of Fruits')
    st.plotly_chart(fig_bar)

# Tab 3: Scatter Plot
with tabs[2]:
    st.header("Scatter Plot Example")
    df_scatter = pd.DataFrame(data_scatter)
    fig_scatter = px.scatter(df_scatter, x='X Values', y='Y Values', title='Scatter Plot Example')
    st.plotly_chart(fig_scatter)
