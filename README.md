# Cell Counts Visualization

This Streamlit application visualizes single cell RNA sequencing data from the perivascular adipose tissue of male and female rats that were fed either a control or high-fat diet for 8 or 24-weeks. Rats fed a high-fat diet for 24-weeks develop hypertension. Included in these visualizations are weight and blood pressure measurements, cell count data, and descriptive statistics for every gene in various cell types grouped by the treatment group these samples were from.

## Plot Explanation

The application allows users to plot cell counts based on selected criteria:

- **Cell Types**: Different types of cells are categorized and can be selected to show their respective counts.
- **Sex**: The data can be filtered by sex (e.g., Male, Female).
- **Time**: Different time points are available to analyze how cell counts change over time.
- **Diet**: Various diet conditions can be selected to see their impact on cell counts.

### How to Use

1. **Login**: Start by logging in with your credentials.
2. **Select Criteria**:
    - **Cell Type**: Choose one or multiple cell types from the sidebar.
    - **Sex**: Filter by sex to see relevant data.
    - **Time**: Select different time points to visualize.
    - **Diet**: Pick diet conditions to filter the data.
3. **View the Plot**: The bar chart will dynamically update based on your selections, showing the cell counts grouped by the selected criteria.

### Bar Chart

The bar chart presents:
- **X-Axis**: Different cell types.
- **Y-Axis**: The count of cells.
- **Color**: Groups the data by a combination of sex, time, and diet for clear comparison.

### Interactive Features

- Hover over the bars to see detailed information about cell counts.
- Use the sidebar to adjust the filters and immediately see the changes reflected in the plot.

This visualization helps in understanding the distribution and impact of different conditions on cell counts effectively.

Enjoy exploring the data!
