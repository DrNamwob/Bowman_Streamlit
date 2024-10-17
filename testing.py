import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time 
from auth import login

if login():
    
    st.header('Cell Counts Figure')
    counts_df = pd.read_csv('grouped_cell_counts.csv')
    counts_data = counts_df[['celltype_broad', 'sex', 'time', 'diet', 'count']]

    # Create a new column that combines 'sex' and 'time' into a single category
    counts_data['group'] = counts_data['sex'] + ' ' + counts_data['time'] + ' ' + counts_data['diet']


    selected_celltype = st.multiselect('Select cell type(s):', options=counts_data['celltype_broad'].unique(), 
                                    default=counts_data['celltype_broad'].unique())

    selected_sex = st.multiselect('Select sex):', options=counts_data['sex'].unique(), 
                                    default=counts_data['sex'].unique())

    selected_time = st.multiselect('Select time):', options=counts_data['time'].unique(), 
                                    default=counts_data['time'].unique())

    selected_diet = st.multiselect('Select diet):', options=counts_data['diet'].unique(), 
                                    default=counts_data['diet'].unique())


    # Filter the data based on user selection
    filtered_data = counts_data[((counts_data['celltype_broad'].isin(selected_celltype)) 
                                & (counts_data['sex'].isin(selected_sex)) 
                                & (counts_data['time'].isin(selected_time))
                                & (counts_data['diet'].isin(selected_diet)))]

    

    # Plotly Bar Chart
    fig = px.bar(
        filtered_data,
        x='celltype_broad',
        y='count',
        color='group',  # Use the new 'group' column to show different groups (Female 8W, Male 8W, etc.)
        barmode='group',  # Group the bars for each 'celltype_broad'
        labels={'count': 'Cell Count', 'celltype_broad': 'Cell Type', 'group': 'Group'},
        #color_discrete_map={'M 24W': 'steelblue', 'M 8W': 'aqua', 'F 24W': 'mediumseagreen', 'F 8W': 'greenyellow'},
        title='Cell Counts by Cell Type, Sex, and Time'
    )

    st.plotly_chart(fig)

    # Weight, Systolic BP, or Mean Arterial Pressure:

    st.header('Physiologic Measurements Figure')
    st.write('Choose between mean and median')

    # Streamlit selectbox for choosing Mean or Median
    statistic = st.selectbox("Select Statistic", ["Mean", "Median"])

    # Streamlit selectbox for choosing the variable to plot
    variable = st.selectbox("Select Variable", ["Weight", "Systolic Blood Pressure", "Mean Arterial Pressure"])



    # Check if the statistic is 'Mean' and the variable is 'mean arterial pressure'
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    # Check if the statistic is 'Mean' and the variable is 'mean arterial pressure'
    if statistic == "Mean":
        if variable == "Mean Arterial Pressure":
            # Load the data
            df = pd.read_csv('map_df_mean.csv')
            
            # Create a new column that combines 'sex' and 'diet' for the X-axis
            df['group'] = df['sex'] + ' ' + df['diet']

            # Let users select columns to plot (systolic_bp columns)
            selected_columns = st.multiselect(
                'Select columns to plot:', 
                options=[col for col in df.columns if 'MAP' in col],  # Only systolic_bp columns
                default=[col for col in df.columns if 'MAP' in col]  # Default to all systolic_bp columns
            )

            # Filter the DataFrame based on selected columns
            df_selected = df[['group', 'sex', 'diet'] + selected_columns]  # Keep 'group', 'sex', and 'diet'

            # Melt the DataFrame for easier plotting (long format)
            df_melted = df_selected.melt(id_vars=['group', 'sex', 'diet'], 
                                        value_vars=selected_columns, 
                                        var_name='Time', 
                                        value_name='Mean Arterial Pressure')

            # Plotly Bar Chart
            fig = px.bar(
                df_melted,
                x='Time',  # The X-axis will show the selected systolic_bp columns
                y='Mean Arterial Pressure',
                color='group',  # Color by 'group' (sex + diet)
                barmode='group',  # Group the bars for comparison
                labels={'group': 'Group (Sex - Diet)', 'Mean Arterial Pressure': 'Mean Arterial Pressure \n (mmHg)'},
                title=f'{statistic} of {variable.title()} by Group and Week'
            )

            # Display the plot
            st.plotly_chart(fig)
            
        elif variable == 'Systolic Blood Pressure':

            df = pd.read_csv('sbp_df_mean.csv')
            
            # Create a new column that combines 'sex' and 'diet' for the X-axis
            df['group'] = df['sex'] + ' ' + df['diet']

            # Let users select columns to plot (systolic_bp columns)
            selected_columns = st.multiselect(
                'Select columns to plot:', 
                options=[col for col in df.columns if 'systolic_bp' in col],  # Only systolic_bp columns
                default=[col for col in df.columns if 'systolic_bp' in col]  # Default to all systolic_bp columns
            )

            # Filter the DataFrame based on selected columns
            df_selected = df[['group', 'sex', 'diet'] + selected_columns]  # Keep 'group', 'sex', and 'diet'

            # Melt the DataFrame for easier plotting (long format)
            df_melted = df_selected.melt(id_vars=['group', 'sex', 'diet'], 
                                        value_vars=selected_columns, 
                                        var_name='Time', 
                                        value_name='Systolic Blood Pressure')

            # Plotly Bar Chart
            fig = px.bar(
                df_melted,
                x='Time',  # The X-axis will show the selected systolic_bp columns
                y='Systolic Blood Pressure',
                color='group',  # Color by 'group' (sex + diet)
                barmode='group',  # Group the bars for comparison
                labels={'group': 'Group (Sex - Diet)', 'Systolic Blood Pressure': 'Systolic Blood Pressure \n (mmHg)'},
                title=f'{statistic} of {variable.title()} by Group and Week'
            )

            # Display the plot
            st.plotly_chart(fig)

        else: #variable == 'Weight':
        
            df = pd.read_csv('weights_df_mean.csv')
            
            # Create a new column that combines 'sex' and 'diet' for the X-axis
            df['group'] = df['sex'] + ' ' + df['diet']

            # Let users select columns to plot (systolic_bp columns)
            selected_columns = st.multiselect(
                'Select columns to plot:', 
                options=[col for col in df.columns if 'weight' in col],  # Only systolic_bp columns
                default=[col for col in df.columns if 'weight' in col]  # Default to all systolic_bp columns
            )

            # Filter the DataFrame based on selected columns
            df_selected = df[['group', 'sex', 'diet'] + selected_columns]  # Keep 'group', 'sex', and 'diet'

            # Melt the DataFrame for easier plotting (long format)
            df_melted = df_selected.melt(id_vars=['group', 'sex', 'diet'], 
                                        value_vars=selected_columns, 
                                        var_name='Time', 
                                        value_name='Weight')

            # Plotly Bar Chart
            fig = px.bar(
                df_melted,
                x='Time',  # The X-axis will show the selected systolic_bp columns
                y='Weight',
                color='group',  # Color by 'group' (sex + diet)
                barmode='group',  # Group the bars for comparison
                labels={'group': 'Group (Sex - Diet)', 'Weight': 'Weight (g)'},
                title=f'{statistic} of {variable.title()} by Group and Week'
            )

            # Display the plot
            st.plotly_chart(fig)

    if statistic == "Median":
        if variable == "Mean Arterial Pressure":
            # Load the data
            df = pd.read_csv('map_df_median.csv')
            
            # Create a new column that combines 'sex' and 'diet' for the X-axis
            df['group'] = df['sex'] + ' ' + df['diet']

            # Let users select columns to plot (systolic_bp columns)
            selected_columns = st.multiselect(
                'Select columns to plot:', 
                options=[col for col in df.columns if 'MAP' in col],  # Only systolic_bp columns
                default=[col for col in df.columns if 'MAP' in col]  # Default to all systolic_bp columns
            )

            # Filter the DataFrame based on selected columns
            df_selected = df[['group', 'sex', 'diet'] + selected_columns]  # Keep 'group', 'sex', and 'diet'

            # Melt the DataFrame for easier plotting (long format)
            df_melted = df_selected.melt(id_vars=['group', 'sex', 'diet'], 
                                        value_vars=selected_columns, 
                                        var_name='Time', 
                                        value_name='Mean Arterial Pressure')

            # Plotly Bar Chart
            fig = px.bar(
                df_melted,
                x='Time',  # The X-axis will show the selected systolic_bp columns
                y='Mean Arterial Pressure',
                color='group',  # Color by 'group' (sex + diet)
                barmode='group',  # Group the bars for comparison
                labels={'group': 'Group (Sex - Diet)', 'Mean Arterial Pressure': 'Mean Arterial Pressure \n (mmHg)'},
                title=f'{statistic} of {variable.title()} by Group and Week'
            )

            # Display the plot
            st.plotly_chart(fig)
            
        elif variable == 'Systolic Blood Pressure':

            df = pd.read_csv('sbp_df_median.csv')
            
            # Create a new column that combines 'sex' and 'diet' for the X-axis
            df['group'] = df['sex'] + ' ' + df['diet']

            # Let users select columns to plot (systolic_bp columns)
            selected_columns = st.multiselect(
                'Select columns to plot:', 
                options=[col for col in df.columns if 'systolic_bp' in col],  # Only systolic_bp columns
                default=[col for col in df.columns if 'systolic_bp' in col]  # Default to all systolic_bp columns
            )

            # Filter the DataFrame based on selected columns
            df_selected = df[['group', 'sex', 'diet'] + selected_columns]  # Keep 'group', 'sex', and 'diet'

            # Melt the DataFrame for easier plotting (long format)
            df_melted = df_selected.melt(id_vars=['group', 'sex', 'diet'], 
                                        value_vars=selected_columns, 
                                        var_name='Time', 
                                        value_name='Systolic Blood Pressure')

            # Plotly Bar Chart
            fig = px.bar(
                df_melted,
                x='Time',  # The X-axis will show the selected systolic_bp columns
                y='Systolic Blood Pressure',
                color='group',  # Color by 'group' (sex + diet)
                barmode='group',  # Group the bars for comparison
                labels={'group': 'Group (Sex - Diet)', 'Systolic Blood Pressure': 'Systolic Blood Pressure \n (mmHg)'},
                title=f'{statistic} of {variable.title()} by Group and Week'
            )

            # Display the plot
            st.plotly_chart(fig)

        else: #variable == 'Weight':
        
            df = pd.read_csv('weights_df_median.csv')
            
            # Create a new column that combines 'sex' and 'diet' for the X-axis
            df['group'] = df['sex'] + ' ' + df['diet']

            # Let users select columns to plot (systolic_bp columns)
            selected_columns = st.multiselect(
                'Select columns to plot:', 
                options=[col for col in df.columns if 'weight' in col],  # Only systolic_bp columns
                default=[col for col in df.columns if 'weight' in col]  # Default to all systolic_bp columns
            )

            # Filter the DataFrame based on selected columns
            df_selected = df[['group', 'sex', 'diet'] + selected_columns]  # Keep 'group', 'sex', and 'diet'

            # Melt the DataFrame for easier plotting (long format)
            df_melted = df_selected.melt(id_vars=['group', 'sex', 'diet'], 
                                        value_vars=selected_columns, 
                                        var_name='Time', 
                                        value_name='Weight')

            # Plotly Bar Chart
            fig = px.bar(
                df_melted,
                x='Time',  # The X-axis will show the selected systolic_bp columns
                y='Weight',
                color='group',  # Color by 'group' (sex + diet)
                barmode='group',  # Group the bars for comparison
                labels={'group': 'Group (Sex - Diet)', 'Weight': 'Weight (g)'},
                title=f'{statistic} of {variable.title()} by Group and Week'
            )

            # Display the plot
            st.plotly_chart(fig)





    # Cache the data loading function
    @st.cache_data
    def load_data():
        # Use pandas to read the CSV file
        df = pd.read_csv('genes_time_diet_sex_described.csv', header=[0, 1], index_col=[0, 1])
        return df

    # Sidebar checkbox for loading gene data
    load_gene_data = st.sidebar.checkbox("Do you want to load Gene data? This may take a while.")

    if load_gene_data:
        # Display a spinner while loading the CSV
        with st.spinner("Loading gene data... Please wait."):
            df = load_data()
            print('completed loading')

        
        st.header('Visualize descriptive Statistics for Genes of Interest')
        # Sidebar for user selections
        st.sidebar.header('User Selections')

        # Extract unique cell types and time_diet_sex groups for selection
        celltypes = df.index.get_level_values('celltype_broad').unique()
        time_diet_sex_groups = df.index.get_level_values('time_diet_sex').unique()
        gene_names = df.columns.get_level_values(0).unique()

        # Sidebar selections for cell types and time_diet_sex groups
        selected_celltype = st.sidebar.selectbox('Select Cell Type:', options=celltypes)
        

        # Select genes
        st.sidebar.header('Select Gene and Statistic')
        selected_gene = st.sidebar.selectbox('Select Gene:', options=gene_names, index=gene_names.tolist().index('Cebpb'))
        

        # Select descriptive statistic (assuming the second level of columns are the statistics)
        statistics = df[selected_gene].columns.tolist()
        selected_statistic = st.sidebar.selectbox('Select Statistic:', options=statistics)

        # Filter the DataFrame based on user selections
        filtered_df = df.loc[(slice(None), selected_celltype), (selected_gene, selected_statistic)]
        print(f'Filtered dataframe: {filtered_df}')

        # Prepare the DataFrame for plotting
        plot_data = filtered_df.reset_index()
        plot_data.columns = ['time_diet_sex', 'celltype_broad', selected_statistic]

        # Plotly Bar Chart
        fig = px.bar(
            plot_data,
            x='time_diet_sex',
            y=selected_statistic,
            color='time_diet_sex',
            title=f'{selected_gene} ({selected_statistic}) Expression in {selected_celltype} by Time/Diet/Sex',
            labels={'time_diet_sex': 'Time/Diet/Sex', selected_statistic: 'Expression Level'},
            barmode='group'
        )

        # Display the plot
        st.plotly_chart(fig)

        # Optional: Display the filtered DataFrame
        st.write("Filtered DataFrame:")
        st.dataframe(plot_data)
    else:
        st.write("Gene data is not loaded. Please check the box to load data.")











