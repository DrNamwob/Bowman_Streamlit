import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
from auth import login
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, root_mean_squared_error


st.set_page_config(page_title="My Multipage App", layout="wide")


if login():
    # Define the pages using classes
    class HomePage:
        def layout(self):
            st.title("Welcome to My App")

            # Read the contents of README.md
            with open("README.md", "r") as file:
                readme_contents = file.read()

            # Display the contents of README.md
            st.markdown(readme_contents)

    class CellCountsPage:
        def layout(self):
            st.header("Cell Counts Figure")
            counts_df = pd.read_csv("grouped_cell_counts.csv")
            counts_data = counts_df[["celltype_broad", "sex", "time", "diet", "count"]]
            counts_data["group"] = (
                counts_data["sex"]
                + " "
                + counts_data["time"]
                + " "
                + counts_data["diet"]
            )

            selected_celltype = st.multiselect(
                "Select cell type(s):",
                options=counts_data["celltype_broad"].unique(),
                default=counts_data["celltype_broad"].unique(),
            )
            selected_sex = st.multiselect(
                "Select sex:",
                options=counts_data["sex"].unique(),
                default=counts_data["sex"].unique(),
            )
            selected_time = st.multiselect(
                "Select time:",
                options=counts_data["time"].unique(),
                default=counts_data["time"].unique(),
            )
            selected_diet = st.multiselect(
                "Select diet:",
                options=counts_data["diet"].unique(),
                default=counts_data["diet"].unique(),
            )

            filtered_data = counts_data[
                (
                    (counts_data["celltype_broad"].isin(selected_celltype))
                    & (counts_data["sex"].isin(selected_sex))
                    & (counts_data["time"].isin(selected_time))
                    & (counts_data["diet"].isin(selected_diet))
                )
            ]

            fig = px.bar(
                filtered_data,
                x="celltype_broad",
                y="count",
                color="group",
                barmode="group",
                labels={
                    "count": "Cell Count",
                    "celltype_broad": "Cell Type",
                    "group": "Group",
                },
                title="Cell Counts by Cell Type, Sex, and Time",
            )
            st.plotly_chart(fig)

    class SecondFigurePage:
        def layout(self):
            st.header("Physiological Data Figure")

            st.header("Physiologic Measurements Figure")
            st.write("Choose between mean and median")

            # Streamlit selectbox for choosing Mean or Median
            statistic = st.selectbox("Select Statistic", ["Mean", "Median"])

            # Streamlit selectbox for choosing the variable to plot
            variable = st.selectbox(
                "Select Variable",
                [
                    "Pulse Wave Velocity",
                    "Weight",
                    "Systolic Blood Pressure",
                    "Mean Arterial Pressure",
                ],
            )
            width = 2000
            height = 1000
            # Check if the statistic is 'Mean' and the variable is 'Pulse Wave Velocity'
            if statistic == "Mean":
                if variable == "Pulse Wave Velocity":
                    # Assuming df_combined is already defined and contains the necessary data
                    df_combined = pd.read_csv("pwv_combined.csv")

                    # Create a bar plot using Plotly Express
                    fig = px.line(
                        df_combined,
                        x="Time_on_Diet",
                        y="mean",
                        color="group",
                        title=f"{statistic} of Pulse Wave Velocity over Time",
                        labels={
                            "mean": "Pulse Wave Velocity",
                            "Time_on_Diet": "Time on Diet (weeks)",
                            "group": "Group",
                        },
                        width=width,
                        height=height,
                    )

                    # Display the plot in Streamlit
                    st.plotly_chart(fig)

                elif variable == "Mean Arterial Pressure":
                    # Load the data
                    df = pd.read_csv("map_df_mean.csv")

                    # Create a new column that combines 'sex' and 'diet' for the X-axis
                    df["group"] = df["sex"] + " " + df["diet"]

                    # Let users select columns to plot (MAP columns)
                    selected_columns = st.multiselect(
                        "Select columns to plot:",
                        options=[
                            col for col in df.columns if "MAP" in col
                        ],  # Only MAP columns
                        default=[
                            col for col in df.columns if "MAP" in col
                        ],  # Default to all MAP columns
                        key="map_columns",  # Unique key for this multiselect
                    )

                    # Filter the DataFrame based on selected columns
                    df_selected = df[
                        ["group", "sex", "diet"] + selected_columns
                    ]  # Keep 'group', 'sex', and 'diet'

                    # Melt the DataFrame for easier plotting (long format)
                    df_melted = df_selected.melt(
                        id_vars=["group", "sex", "diet"],
                        value_vars=selected_columns,
                        var_name="Time",
                        value_name="Mean Arterial Pressure",
                    )

                    # Plotly Bar Chart
                    fig = px.bar(
                        df_melted,
                        x="Time",  # The X-axis will show the selected MAP columns
                        y="Mean Arterial Pressure",
                        color="group",  # Color by 'group' (sex + diet)
                        barmode="group",  # Group the bars for comparison
                        labels={
                            "group": "Group (Sex - Diet)",
                            "Mean Arterial Pressure": "Mean Arterial Pressure \n (mmHg)",
                        },
                        title=f"{statistic} of {variable.title()} by Group and Week",
                        width=width,
                        height=height,
                    )

                    # Display the plot
                    st.plotly_chart(fig)

                elif variable == "Systolic Blood Pressure":
                    df = pd.read_csv("sbp_df_mean.csv")

                    # Create a new column that combines 'sex' and 'diet' for the X-axis
                    df["group"] = df["sex"] + " " + df["diet"]

                    # Let users select columns to plot (systolic_bp columns)
                    selected_columns = st.multiselect(
                        "Select columns to plot:",
                        options=[
                            col for col in df.columns if "systolic_bp" in col
                        ],  # Only systolic_bp columns
                        default=[
                            col for col in df.columns if "systolic_bp" in col
                        ],  # Default to all systolic_bp columns
                        key="sbp_columns",  # Unique key for this multiselect
                    )

                    # Filter the DataFrame based on selected columns
                    df_selected = df[
                        ["group", "sex", "diet"] + selected_columns
                    ]  # Keep 'group', 'sex', and 'diet'

                    # Melt the DataFrame for easier plotting (long format)
                    df_melted = df_selected.melt(
                        id_vars=["group", "sex", "diet"],
                        value_vars=selected_columns,
                        var_name="Time",
                        value_name="Systolic Blood Pressure",
                    )

                    # Plotly Bar Chart
                    fig = px.bar(
                        df_melted,
                        x="Time",  # The X-axis will show the selected systolic_bp columns
                        y="Systolic Blood Pressure",
                        color="group",  # Color by 'group' (sex + diet)
                        barmode="group",  # Group the bars for comparison
                        labels={
                            "group": "Group (Sex - Diet)",
                            "Systolic Blood Pressure": "Systolic Blood Pressure \n (mmHg)",
                        },
                        title=f"{statistic} of {variable.title()} by Group and Week",
                        width=width,
                        height=height,
                    )

                    # Display the plot
                    st.plotly_chart(fig)

                else:  # variable == 'Weight'
                    df = pd.read_csv("weights_df_mean.csv")

                    # Create a new column that combines 'sex' and 'diet' for the X-axis
                    df["group"] = df["sex"] + " " + df["diet"]

                    # Let users select columns to plot (weight columns)
                    selected_columns = st.multiselect(
                        "Select columns to plot:",
                        options=[
                            col for col in df.columns if "weight" in col
                        ],  # Only weight columns
                        default=[
                            col for col in df.columns if "weight" in col
                        ],  # Default to all weight columns
                        key="weight_columns",  # Unique key for this multiselect
                    )

                    # Filter the DataFrame based on selected columns
                    df_selected = df[
                        ["group", "sex", "diet"] + selected_columns
                    ]  # Keep 'group', 'sex', and 'diet'

                    # Melt the DataFrame for easier plotting (long format)
                    df_melted = df_selected.melt(
                        id_vars=["group", "sex", "diet"],
                        value_vars=selected_columns,
                        var_name="Time",
                        value_name="Weight",
                    )

                    # Plotly Bar Chart
                    fig = px.bar(
                        df_melted,
                        x="Time",  # The X-axis will show the selected weight columns
                        y="Weight",
                        color="group",  # Color by 'group' (sex + diet)
                        barmode="group",  # Group the bars for comparison
                        labels={"group": "Group (Sex - Diet)", "Weight": "Weight (g)"},
                        title=f"{statistic} of {variable.title()} by Group and Week",
                        width=width,
                        height=height,
                    )

                    # Display the plot
                    st.plotly_chart(fig)

            if statistic == "Median":
                if variable == "Pulse Wave Velocity":
                    # Assuming df_combined is already defined and contains the necessary data
                    df_combined = pd.read_csv("pwv_combined.csv")

                    # Create a bar plot using Plotly Express
                    fig = px.line(
                        df_combined,
                        x="Time_on_Diet",
                        y="50%",
                        color="group",
                        title=f"{statistic} of Pulse Wave Velocity over Time",
                        labels={
                            "50%": "Pulse Wave Velocity",
                            "Time_on_Diet": "Time on Diet (weeks)",
                            "group": "Group",
                        },
                        width=width,
                        height=height,
                    )

                    # Display the plot in Streamlit
                    st.plotly_chart(fig)

                elif variable == "Mean Arterial Pressure":
                    # Load the data
                    df = pd.read_csv("map_df_median.csv")

                    # Create a new column that combines 'sex' and 'diet' for the X-axis
                    df["group"] = df["sex"] + " " + df["diet"]

                    # Let users select columns to plot (MAP columns)
                    selected_columns = st.multiselect(
                        "Select columns to plot:",
                        options=[
                            col for col in df.columns if "MAP" in col
                        ],  # Only MAP columns
                        default=[
                            col for col in df.columns if "MAP" in col
                        ],  # Default to all MAP columns
                        key="map_columns_median",  # Unique key for this multiselect
                    )

                    # Filter the DataFrame based on selected columns
                    df_selected = df[
                        ["group", "sex", "diet"] + selected_columns
                    ]  # Keep 'group', 'sex', and 'diet'

                    # Melt the DataFrame for easier plotting (long format)
                    df_melted = df_selected.melt(
                        id_vars=["group", "sex", "diet"],
                        value_vars=selected_columns,
                        var_name="Time",
                        value_name="Mean Arterial Pressure",
                    )

                    # Plotly Bar Chart
                    fig = px.bar(
                        df_melted,
                        x="Time",  # The X-axis will show the selected MAP columns
                        y="Mean Arterial Pressure",
                        color="group",  # Color by 'group' (sex + diet)
                        barmode="group",  # Group the bars for comparison
                        labels={
                            "group": "Group (Sex - Diet)",
                            "Mean Arterial Pressure": "Mean Arterial Pressure \n (mmHg)",
                        },
                        title=f"{statistic} of {variable.title()} by Group and Week",
                        width=width,
                        height=height,
                    )

                    # Display the plot
                    st.plotly_chart(fig)

                elif variable == "Systolic Blood Pressure":
                    df = pd.read_csv("sbp_df_median.csv")

                    # Create a new column that combines 'sex' and 'diet' for the X-axis
                    df["group"] = df["sex"] + " " + df["diet"]

                    # Let users select columns to plot (systolic_bp columns)
                    selected_columns = st.multiselect(
                        "Select columns to plot:",
                        options=[
                            col for col in df.columns if "systolic_bp" in col
                        ],  # Only systolic_bp columns
                        default=[
                            col for col in df.columns if "systolic_bp" in col
                        ],  # Default to all systolic_bp columns
                        key="sbp_columns_median",  # Unique key for this multiselect
                    )

                    # Filter the DataFrame based on selected columns
                    df_selected = df[
                        ["group", "sex", "diet"] + selected_columns
                    ]  # Keep 'group', 'sex', and 'diet'

                    # Melt the DataFrame for easier plotting (long format)
                    df_melted = df_selected.melt(
                        id_vars=["group", "sex", "diet"],
                        value_vars=selected_columns,
                        var_name="Time",
                        value_name="Systolic Blood Pressure",
                    )

                    # Plotly Bar Chart
                    fig = px.bar(
                        df_melted,
                        x="Time",  # The X-axis will show the selected systolic_bp columns
                        y="Systolic Blood Pressure",
                        color="group",  # Color by 'group' (sex + diet)
                        barmode="group",  # Group the bars for comparison
                        labels={
                            "group": "Group (Sex - Diet)",
                            "Systolic Blood Pressure": "Systolic Blood Pressure \n (mmHg)",
                        },
                        title=f"{statistic} of {variable.title()} by Group and Week",
                        width=width,
                        height=height,
                    )

                    # Display the plot
                    st.plotly_chart(fig)

                else:  # variable == 'Weight'
                    df = pd.read_csv("weights_df_median.csv")

                    # Create a new column that combines 'sex' and 'diet' for the X-axis
                    df["group"] = df["sex"] + " " + df["diet"]

                    # Let users select columns to plot (weight columns)
                    selected_columns = st.multiselect(
                        "Select columns to plot:",
                        options=[
                            col for col in df.columns if "weight" in col
                        ],  # Only weight columns
                        default=[
                            col for col in df.columns if "weight" in col
                        ],  # Default to all weight columns
                        key="weight_columns_median",  # Unique key for this multiselect
                    )

                    # Filter the DataFrame based on selected columns
                    df_selected = df[
                        ["group", "sex", "diet"] + selected_columns
                    ]  # Keep 'group', 'sex', and 'diet'

                    # Melt the DataFrame for easier plotting (long format)
                    df_melted = df_selected.melt(
                        id_vars=["group", "sex", "diet"],
                        value_vars=selected_columns,
                        var_name="Time",
                        value_name="Weight",
                    )

                    # Plotly Bar Chart
                    fig = px.bar(
                        df_melted,
                        x="Time",  # The X-axis will show the selected weight columns
                        y="Weight",
                        color="group",  # Color by 'group' (sex + diet)
                        barmode="group",  # Group the bars for comparison
                        labels={"group": "Group (Sex - Diet)", "Weight": "Weight (g)"},
                        title=f"{statistic} of {variable.title()} by Group and Week",
                        width=width,
                        height=height,
                    )

                    # Display the plot
                    st.plotly_chart(fig)
                    # Add the code for your second figure here

    class ThirdFigurePage:
        def layout(self):
            st.header("Gene Expression Plot")

            # Cache the data loading function
            @st.cache_data
            def load_data():
                # Use pandas to read the CSV file
                df = pd.read_csv(
                    "genes_time_diet_sex_described.csv", header=[0, 1], index_col=[0, 1]
                )
                return df

            # Sidebar checkbox for loading gene data
            load_gene_data = st.checkbox(
                "Do you want to load Gene data? This may take a while. ~1 - 2 minutes"
            )

            if load_gene_data:
                # Display a spinner while loading the CSV
                with st.spinner("Loading gene data... Please wait."):
                    df = load_data()
                    print("completed loading")

                st.header("Visualize descriptive Statistics for Genes of Interest")
                # Sidebar for user selections
                st.sidebar.header("User Selections")

                # Extract unique cell types and time_diet_sex groups for selection
                celltypes = df.index.get_level_values("celltype_broad").unique()
                time_diet_sex_groups = df.index.get_level_values(
                    "time_diet_sex"
                ).unique()
                gene_names = df.columns.get_level_values(0).unique()

                # Sidebar selections for cell types and time_diet_sex groups
                selected_celltype = st.sidebar.selectbox(
                    "Select Cell Type:", options=celltypes
                )

                # Select genes
                st.sidebar.header("Select Gene and Statistic")
                selected_gene = st.sidebar.selectbox(
                    "Select Gene:",
                    options=gene_names,
                    index=gene_names.tolist().index("Cebpb"),
                )

                # Select descriptive statistic (assuming the second level of columns are the statistics)
                statistics = df[selected_gene].columns.tolist()
                selected_statistic = st.sidebar.selectbox(
                    "Select Statistic:", options=statistics
                )

                # Filter the DataFrame based on user selections
                filtered_df = df.loc[
                    (slice(None), selected_celltype),
                    (selected_gene, selected_statistic),
                ]
                print(f"Filtered dataframe: {filtered_df}")

                # Prepare the DataFrame for plotting
                plot_data = filtered_df.reset_index()
                plot_data.columns = [
                    "time_diet_sex",
                    "celltype_broad",
                    selected_statistic,
                ]

                # Plotly Bar Chart
                fig = px.bar(
                    plot_data,
                    x="time_diet_sex",
                    y=selected_statistic,
                    color="time_diet_sex",
                    title=f"{selected_gene} ({selected_statistic}) Expression in {selected_celltype} by Time/Diet/Sex",
                    labels={
                        "time_diet_sex": "Time/Diet/Sex",
                        selected_statistic: "Expression Level",
                    },
                    barmode="group",
                )

                # Display the plot
                st.plotly_chart(fig)

                # Optional: Display the filtered DataFrame
                st.write("Filtered DataFrame:")
                st.dataframe(plot_data)
            else:
                st.write("Gene data is not loaded. Please check the box to load data.")
                # Add the code for your second figure here

    class FourthPage:
        def layout(self):
            st.header("Machine Learning Model Training")

            def countdown(seconds):
                countdown_placeholder = st.empty()
                for i in range(seconds, 0, -1):
                    countdown_placeholder.write(
                        f"Begin model training in {i} seconds..."
                    )
                    time.sleep(1)
                countdown_placeholder.write("Starting model training now...")

            @st.cache_data
            def load_data():
                # Use pandas to read the CSV file
                df = pd.read_csv(
                    "focal_adhesion_pi3k_genes_only_unscaled.csv"
                )
                return df

            load_focal_adhesion_data = st.checkbox(
                "Do you want to load the data? This may take a while. ~ 1 - 2 minutes"
            )

            if load_focal_adhesion_data:
                # Display a spinner while loading the CSV
                with st.spinner("Loading gene data... Please wait."):
                    df = load_data()
                    df.set_index("Unnamed: 0", inplace=True)
         
                
                train_model_question = st.checkbox(
                        "Do you want to train the models to predict pulse wave velocity? This may take a while. ~5 minutes"
                    )
           

                if train_model_question:
                    st.header("Select which models to train and which scaler(s) to use.")
                    select_models = st.multiselect(
                        "Select models to train:",
                        [
                            "Linear Regression",
                            "Random Forest Regressor",
                            "Ridge Regression",
                            "Lasso Regression",
                            "K-Nearest Neighbors Regressor",
                        ],
                    )
                    
                    select_scalers = st.multiselect(
                        "Select scalers to use:",
                        ["Standard Scaler", "Min-Max Scaler"],
                    )
                    
                    
                    proceed_question = st.checkbox("Proceed with training?")
                    if proceed_question:
                        countdown(5)
                        with st.spinner("Training... Please wait."):
                            y = df.pop("PWV")
                            X = df

                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, random_state=42
                            )
                            scalers = {'Standard Scaler': StandardScaler(), 'Min-Max Scaler': MinMaxScaler()}

                            models = {
                                "Linear Regression": LinearRegression(),
                                "Random Forest Regressor": RandomForestRegressor(
                                    random_state=42
                                ),
                                "Ridge Regression": Ridge(),
                                "Lasso Regression": Lasso(),
                                "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
                            }
                            
                            selected_scalers = {name: scaler for name, scaler in scalers.items() if name in select_scalers}
                            selected_models = {name: model for name, model in models.items() if name in select_models}
                            for scaler_name, scaler in selected_scalers.items():
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                st.write(f"Using {scaler_name} as the scaler.")
                                st.write("-" * 30)
                                # Train and evaluate each model
                                for model_name, model in selected_models.items():
                                    st.write(f"Training {model_name}... with {scaler_name}")

                                    # Train the model
                                    model.fit(X_train_scaled, y_train)

                                    # Predict on the test set
                                    y_pred = model.predict(X_test_scaled)

                                    # Evaluate the model
                                    rmse = root_mean_squared_error(y_test, y_pred)
                                    r2 = r2_score(y_test, y_pred)

                                    st.write(f"{model_name} Results:")
                                    st.write(f"Root Mean Squared Error: {rmse:.4f}")
                                    st.write(f"R^2 Score: {r2:.4f}")
                                    st.write("-" * 30)
    pages = {
        "Home": HomePage,
        "Dynamic Cell Counts by Group Figure": CellCountsPage,
        "Dynamic Physiologic Parameters Figure": SecondFigurePage,
        "Dynamic Gene by Cell Type Figure": ThirdFigurePage,
        "Training ML Models": FourthPage,
    }

    # Display navigation
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to:", list(pages.keys()))

    # Display the selected page
    page = pages[selection]()
    page.layout()
