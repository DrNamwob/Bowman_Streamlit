
This Streamlit application visualizes single cell RNA sequencing (scRNAseq) data from the perivascular adipose tissue (PVAT) of male and female rats that were fed either a control or high-fat diet for 8 or 24 weeks. Rats fed a high-fat diet for 24 weeks develop hypertension. Included in these visualizations are weight and blood pressure measurements, cell count data, and descriptive statistics for every gene in various cell types grouped by the treatment group these samples were from.

## The long-term goal for this project is to use the gene expression data from this dataset to predict the pulse wave velocity of these animals.


### Pulse wave velocity is an indicator of compliance, or stiffness, of the vasculature. Higher pulse wave velocities indicates stiffer, lower compliance, vessels. Reduced compliance often precedes hypertension (high blood pressure) and is a risk factor for stroke.

- The pulse wave velocity of rats on the high-fat diet markedly increase after about 8 weeks on diet. 
- Understanding which genes (features) have high weights in these predictive models could indicate specific biological pathways that are important in this process. 

## Objectives for this application:

1. Help users explore the complicated scRNAseq data from the PVAT from rats fed either a high-fat diet or "control" diet. This includes not only the gene expression data from various cell types, but also the animal-level physiological parameters. These include blood pressure, weight, and pulse wave velocity.
2. Develop an intuition for how different machine learning models perform on predicting pulse wave velocity from the gene expression data of individual cells. 
 


The scRNAseq data was processed in the following manner:

# Processing code:

```python
sc.pp.filter_genes(adata, min_counts=10) # Removed any genes with less than 10 counts
sc.pp.filter_genes(adata, min_cells=20) # Removed any genes that did not appear in at least 20 cells
sc.pp.filter_cells(adata, min_genes=50) # Removed cells that did not have at least 50 genes
sc.pp.filter_cells(adata, min_counts=100) # Removed cells that did not have at least 100 counts
sc.pp.normalize_total(adata) # Normalize each cell by the median number of counts from the entire dataset. Each cell will have the same number of total counts (equal to the median).
```


There is no missing data per say, because gene counts of 0 are meaningful. Some cells (even of the same type) will have certain genes expressed, while other cells won't. 

## Cell Counts Plot

1. **Select Criteria**:
    - **Cell Type**: Choose one or multiple cell types from the sidebar.
    - **Sex**: Filter by sex to see relevant data.
    - **Time**: Select different time points to visualize.
    - **Diet**: Pick diet conditions to filter the data.
2. **View the Plot**: The bar chart will dynamically update based on your selections, showing the cell counts grouped by the selected criteria.

    ### Bar Chart

    The bar chart presents:
    - **X-Axis**: Different cell types.
    - **Y-Axis**: The count of cells.
    - **Color**: Groups the data by a combination of sex, time, and diet for clear comparison.

## Physiological Data Plot

1. **Select Statistic**: Choose between `Mean` and `Median`.
2. **Select Variable**: Options include `Weight`, `Systolic Blood Pressure`, and `Mean Arterial Pressure`.
3. **Load Data**: Depending on the selected variable, relevant data will be loaded.
4. **Select Columns**: For `Mean Arterial Pressure`, select specific columns to plot.

    ### Bar Chart

    The bar chart for physiological data presents:
    - **X-Axis**: Selected columns (e.g., different time points).
    - **Y-Axis**: Mean or median values of the selected variable.
    - **Color**: Groups the data by a combination of sex and diet for clear comparison.

## Gene Expression Plot

1. **Select Gene**: Choose a gene to visualize.
2. **Select Statistic**: Pick a descriptive statistic (e.g., mean, median) for the gene.
3. **Load Data**: A sidebar checkbox allows users to decide whether to load the gene data.
    ```python
    load_gene_data = st.sidebar.checkbox("Do you want to load Gene data? This may take a while.")
    if load_gene_data:
        with st.spinner("Loading gene data... Please wait."):
            df = load_data()
    ```

    ### Bar Chart

    The bar chart for gene expression presents:
    - **X-Axis**: Different time/diet/sex groups.
    - **Y-Axis**: Expression level of the selected gene.
    - **Color**: Groups the data by time/diet/sex for detailed comparison.

## Data Loading

The application uses caching to optimize data loading:
- **Cache Data Loading**: The `load_data` function is cached to prevent reloading the CSV file every time the app runs.
    ```python
    @st.cache_data
    def load_data():
        df = pd.read_csv('genes_time_diet_sex_described.csv', header=[0, 1], index_col=[0, 1])
        return df
    ```

## Interactive Features

- Hover over the bars to see detailed information.
- Use the sidebar to adjust filters and immediately see the changes reflected in the plot.

This visualization helps in understanding the distribution and impact of different conditions on cell counts and physiological data.

Enjoy exploring the data!
