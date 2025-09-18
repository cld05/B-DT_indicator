import ternary
import mpltern
import eurostat
import json

import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.cm as cm 
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from text_to_print import description_text_by_quarter, description_text_by_countries, load_md_introduction, load_md_methodology, load_md_howto, load_md_welcome, load_md_box_plot
from data_processing import process_import_data, process_ICT_labour_import_data
from utils import debug_print, info_print, error_print
from data_rendering import css

# Set the page configuration at the top of the script
st.set_page_config(
    page_title="B&DT Club Digital Transformation Indicator",  # Optional: Give your app a title
    layout="centered"  # Using the centered layout
)

# Inject custom CSS to control the width of the centered layout
st.markdown(css['body'], unsafe_allow_html=True)

st.warning("**The DTPI is in Beta version**. As such, things might be subject to change.", icon="⚠️")

# Sidebar for navigation
page1 = 'Home'
page2 = "Intro: DTPI"
page3 = "Overview of EU27 DTPI"
page4 = "Zoom into EU27 and EU6 DTPI"

#st.title("Navigation")  # Title for the navigation bar
page = st.radio('Select a page',[page1, page2, page3, page4], horizontal=True, label_visibility='hidden')
st.html(css['logo'])
st.logo(image="logo/DTPI_logo_v5.png")

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()

# weights for the index
w1 = w2 = w3 = 1  # weights for index calc. w1 - GVA, w2 - Employment, w3 - Labour Demand 

# moving average window defined by slider
#window = st.sidebar.slider("Select moving average window", 1, 5, 2)  # Slider to select the window size
#ste window to a fixed value
window = 3

# List of countries for which to process and plot data
# List of countries and titles
countries = ['EU27_2020', 'IT', 'FR', 'DE', 'ES', 'NL', 'SE']
country_titles = ['Europe 27 (EU27)', 'Italy (IT)', 'France (FR)', 'Germany (DE)', 'Spain (ES)', 'Netherlands (NL)', 'Sweden (SE)']
data_to_import = ['GVA', 'employment', 'labour_demand']
#countries = ['IT', 'FR', 'DE']  # Italy, France, and Germany

# Caching data to save time in loading data from API call
@st.cache_data
def load_data():
    # Load the raw data from Eurostat API
    GVA_data_import = eurostat.get_data_df('namq_10_a10')
    info_print("Got GVA data")
    Employment_data_import = eurostat.get_data_df('namq_10_a10_e')
    info_print("Got Employment data")
    Labour_demand_ICT_data_import = eurostat.get_data_df('isoc_sk_oja1')
    info_print("Got Labour Demand data")
    
    # Define the starting quarter for filtering data
    date_start = '2019Q4'

    # Process the raw data
    GVA_data = process_import_data(GVA_data_import, date_start)
    Employment_data = process_import_data(Employment_data_import, date_start)
    Labour_demand_ICT_data = process_ICT_labour_import_data(Labour_demand_ICT_data_import, date_start)

   

    return GVA_data, Employment_data, Labour_demand_ICT_data

with st.spinner("Please wait, loading data..."):
    # Load the data from the cached function
    GVA_data, Employment_data, Labour_demand_ICT_data = load_data()
    info_print("All data has been loaded")

# Create an empty data frame to hold the hold data during transformation
transformed_data = pd.DataFrame()

# Create an empty list to store individual filtered data frames before merging
loaded_data_list = []

# Iterate over each country in the list of countries
for country in countries:
    # Filter GVA data for the given country, where the sector is 'J', unit is 'PC_GDP', item is 'B1G', and data is not seasonally adjusted
    filtered_data_GVA = GVA_data[(GVA_data['nace_r2'] == 'J') & 
                                 (GVA_data['unit'] == 'PC_GDP') & 
                                 (GVA_data['geo'] == country) & 
                                 (GVA_data['na_item'] == 'B1G') & 
                                 (GVA_data['s_adj'] == 'NSA')].copy()

    # Select only 'quarter' and 'value' columns, rename 'value' to 'GVA_value'
    filtered_data_GVA = filtered_data_GVA[['quarter', 'value']]
    #Handling missing values for GVA
    filtered_data_GVA['value'].fillna(method='bfill', inplace=True)
    
    filtered_data_GVA = filtered_data_GVA.rename(columns={'value': f'{country}_GVA_value'})

    # Filter employment data for the given country, with similar filtering criteria (for employment)
    filtered_data_employment = Employment_data[(Employment_data['nace_r2'] == 'J') & 
                                               (Employment_data['unit'] == 'PC_TOT_PER') & 
                                               (Employment_data['geo'] == country) & 
                                               (Employment_data['na_item'] == 'EMP_DC') & 
                                               (Employment_data['s_adj'] == 'NSA')].copy()

    # Select only 'quarter' and 'value' columns, rename 'value' to 'employment_value'
    filtered_data_employment = filtered_data_employment[['quarter', 'value']]
    #Handling missing values for Employment
    filtered_data_employment['value'].fillna(method='bfill', inplace=True)
    filtered_data_employment = filtered_data_employment.rename(columns={'value': f'{country}_employment_value'})

    # Filter labor demand data for the given country, similar criteria
    filtered_data_labour_demand = Labour_demand_ICT_data[(Labour_demand_ICT_data['geo'] == country) &
                                                         (Labour_demand_ICT_data['unit'] == 'PC')].copy()

    # Select only 'quarter' and 'value' columns, rename 'value' to 'employment_value'
    filtered_data_labour_demand = filtered_data_labour_demand[['quarter', 'value']]
    #Handling missing values for Labour Demand
    filtered_data_labour_demand['value'].fillna(method='ffill', inplace=True)

    filtered_data_labour_demand = filtered_data_labour_demand.rename(columns={'value': f'{country}_labour_demand_value'})

    # Merge GVA data and employment data on the 'quarter' column, keep only common entries (inner join)
    merged_data = pd.merge(filtered_data_GVA, filtered_data_employment, on='quarter', how='inner')

    # Merge the resulting data frame with labor demand data on the 'quarter' column
    merged_data = pd.merge(merged_data, filtered_data_labour_demand, on='quarter', how='inner')
    # Set 'quarter' as the index
    merged_data.set_index('quarter', inplace=True)

    # Append the merged data for this country to the list
    loaded_data_list.append(merged_data)

# Concatenate all merged data frames from the list into one data frame
transformed_data = pd.concat(loaded_data_list, axis=1, join='inner')
transformed_data.index = transformed_data.index.strftime('%y-Q%q')

# list of the measures to use in the loop
data_measures = ['GVA', 'employment', 'labour_demand']

for country in countries:
    for measure in data_measures:
        # Calculate the moving average for the specified measure for each country
        moving_average = transformed_data[f'{country}_{measure}_value'].rolling(window=window).mean()

        # Drop NaN values in the moving average 
        moving_average.dropna(inplace=True)

        # Assign the calculated moving average to the DataFrame
        transformed_data[f'{country}_{measure}_moving_average_value'] = moving_average

        # Normalize the moving average using MinMaxScaler
        moving_average_values = moving_average.values.reshape(-1,1)  # reshape to fit method's requirements
        normalized_values = scaler.fit_transform(moving_average_values)

        # Convert normalized values back to a Series with the correct index
        normalized_moving_average = pd.Series(normalized_values.flatten(), index=moving_average.index)

        # Assign the normalized moving average to a new column in the DataFrame and reindex
        transformed_data[f'{country}_{measure}_normalized_moving_average_value'] = (
            normalized_moving_average.reindex(transformed_data.index)
        )

transformed_data.dropna(inplace=True)

# Initialize an empty DataFrame to hold the index values for all countries
index_data = pd.DataFrame(index=transformed_data.index)

for country in countries:
    # Calculate the index as the sum of normalized values
    index_data[f'{country}'] = (
        w1*transformed_data[f'{country}_GVA_normalized_moving_average_value'] +
        w2*transformed_data[f'{country}_employment_normalized_moving_average_value'] +
        w3*transformed_data[f'{country}_labour_demand_normalized_moving_average_value']
    )/(w1+w2+w3)
    index_data.dropna(inplace=True)

# Set global font size for plots
plt.rcParams.update({'font.size': 12,
                    'figure.facecolor': '#002f6c',  # Background color of the figure
                    'axes.facecolor': '#002f6c',    # Background color of the plot area
                    'axes.edgecolor': '#e5e5e5',    # Border color of the plot
                    'axes.labelcolor': '#e5e5e5',   # Color of the axis labels
                    'xtick.color': '#e5e5e5',       # Color of the x-tick labels
                    'ytick.color': '#e5e5e5',       # Color of the y-tick labels
                    'text.color': '#e5e5e5',        # Default text color
                    'axes.titlecolor': '#e5e5e5',   # Color of the title text
                    'grid.color': '#e5e5e5',        # Color of the grid lines
                     })

# The final DataFrame will automatically handle different lengths because of concatenation
#st.write('Custom gradients (raw and normalized) for Employment, GVA, and Labour Demand across countries')
#st.dataframe(custom_gradients_df)

if page == page1:
    st.success("All datasets loaded successfully", icon="⚙️")
    st.info("Datasets are refreshed quarterly at the source", icon="📬")

    st.title('Home page of the Business and Digital Transformation Club DTPI')
    st.markdown(load_md_welcome(), unsafe_allow_html=True, help=None)

elif page == page2:
    st.info("Datasets are refreshed quarterly at the source", icon="📬")

     # Tab 0 is for the Overview, the rest is for selected countries
    intro = 'What about DTPI?'
    methodology = 'Methodology'
    howto = 'How to read it?'
    tab_intro, tab_methodology, tab_howto = st.tabs([intro, methodology, howto])
     
    with tab_intro:
        st.markdown(f'{load_md_introduction()}', unsafe_allow_html=True, help=None)

    with tab_methodology:
        st.markdown(f'{load_md_methodology()}', unsafe_allow_html=True, help=None)

    with tab_howto:
        st.markdown(f'{load_md_howto()}', unsafe_allow_html=True, help=None)
    
elif page == page3:
    st.info("Datasets are refreshed quarterly at the source", icon="📬")
    st.title("Summarising the DTPI for EU27 with the ability to select and compare")

    countries_withoutEU27 = countries.remove('EU27_2020')
    st.markdown(f'{load_md_box_plot()}', unsafe_allow_html=True, help=None)
    options = st.multiselect("**Default is EU6, select one or more countries to compare...**", countries,placeholder="Choose one or more options", disabled=False, label_visibility="visible")
    if not options:
        options = countries
    
    col1, col2 = st.columns([1,1])

    if isinstance(index_data.index, pd.PeriodIndex):
                  index_data.index = index_data.index.to_timestamp()

    with col1:
        # Show all box plots together for a visual comparison
        fig_all_box, ax_all_box = plt.subplots(figsize=(5, 4), dpi=150)
        ax_all_box.boxplot(index_data['EU27_2020'], patch_artist=True, tick_labels=['EU27'], boxprops=dict(facecolor='lightblue'))

        ax_all_box.set_title('Box Plot of Indicator Data EU27', fontsize=10)
        ax_all_box.set_xlabel('Countries', fontsize=8)
        ax_all_box.set_ylabel('Indicator Value', fontsize=8)
        ax_all_box.grid(True)

        st.pyplot(fig_all_box)

        st.write("**DTPI Indicator for EU27**") 
        fig_index, ax_index = plt.subplots(figsize=(5, 4))  # Adjust figure size
        ax_index.plot(index_data.index, index_data['EU27_2020'], marker='x', label='EU27')
        ax_index.set_title(f'Indicator for EU27', fontsize=12)
        ax_index.set_xlabel('Quarter', fontsize=10)
        ax_index.set_ylabel('Indicator Value', fontsize=10)
        ax_index.grid(True)  # Add grid to the plot
        ax_index.tick_params(axis='x', rotation=45, labelsize=9)
        ax_index.tick_params(axis='y', labelsize=9)
        st.pyplot(fig_index)

    with col2:
        # Show all box plots together for a visual comparison
        fig_all_box, ax_all_box = plt.subplots(figsize=(5, 4), dpi=150)
        ax_all_box.boxplot([index_data[country] for country in options], patch_artist=True, tick_labels=options, boxprops=dict(facecolor='lightblue'))

        ax_all_box.set_title('Box Plot of Indicator Data Across Countries', fontsize=10)
        ax_all_box.set_xlabel('Countries', fontsize=8)
        ax_all_box.set_ylabel('Indicator Value', fontsize=8)
        ax_all_box.grid(True)

        st.pyplot(fig_all_box)

        st.write(f"**DTPI Indicator for selected countries**") 
        fig_index, ax_index = plt.subplots(figsize=(5, 4))  # Adjust figure size
        for country in options:
            ax_index.plot(index_data.index, index_data[f'{country}'], marker='x', label=f'{country}')
            ax_index.set_title(f'Indicator for {options}', fontsize=12)
            ax_index.set_xlabel('Quarter', fontsize=10)
            ax_index.set_ylabel('Indicator Value', fontsize=10)
            ax_index.grid(True)  # Add grid to the plot
            ax_index.tick_params(axis='x', rotation=45, labelsize=9)
            ax_index.tick_params(axis='y', labelsize=9)
            ax_index.legend()
        st.pyplot(fig_index)

    index_data_filtered = index_data[['EU27_2020'] + options]
    index_data_filtered.rename(columns={'quarter': 'Quarter'}, inplace=True)
    st.markdown(index_data_filtered.to_markdown(index=True))

    st.markdown(f'---')
    st.markdown(f'### Historical DTPI analysis and highlights for your selection: {options}')

    # TODO code to be refactored in a renderer function
    highlights_text_by_year = description_text_by_countries()
    years = sorted(highlights_text_by_year.keys(), reverse=True)
    expanded = True
    debug_print('>>> Contents tree')
    for year in years:
        debug_print(f'{year}')
        quarters = sorted(highlights_text_by_year[year].keys(), reverse=True)
        for quarter in quarters:
            details = ''
            debug_print(f'  {quarter}')
            contents = sorted(highlights_text_by_year[year][quarter])
            details += f'<details {"open" if expanded else ""}><summary>EU</summary>{highlights_text_by_year[year][quarter]["EU"]}</details>'
            st.markdown('---', unsafe_allow_html=True, help=None)
            for content in contents:
                if content in options:
                    debug_print(f'    {content}')
                    details += f'<details {"open" if expanded else ""}><summary>{content}</summary>{highlights_text_by_year[year][quarter][content]}</details>'
            if expanded:
                st.markdown(f'<details open><summary>{year} {quarter}</summary>{details}</details>', unsafe_allow_html=True, help=None)
                expanded = not expanded
            else:
                st.markdown(f'<details><summary>{year} {quarter}</summary>{details}</details>', unsafe_allow_html=True, help=None)

elif page == page4:
     st.info("Datasets are refreshed quarterly at the source", icon="📬")
     st.title("Zooming into the EU27 and EU6 components of the DTPI")

     # Tab 0 is for the Overview, the rest is for selected countries
     tabs = st.tabs([f'{title}' for title in country_titles])
     tab_idx = 0
     for idx, country in enumerate(countries):
         
         with tabs[tab_idx]:
             
             st.markdown(f'### Data for **{country_titles[idx]}**: you can scroll and zoom into the details for the different views')
             
             col1, col2 = st.columns([1,2])
             if isinstance(transformed_data.index, pd.PeriodIndex):
                    transformed_data.index = transformed_data.index.to_timestamp()
            
             if isinstance(index_data.index, pd.PeriodIndex):
                    index_data.index = index_data.index.to_timestamp()

            # Column 1 content: ICT Employment, GVA, and Labour Demand Data
             with col1:
                st.write("**ICT Employment Data**")
                # Ensure the index is only converted if it's a PeriodIndex
                fig1, ax1 = plt.subplots(figsize=(4, 2.5))  # Adjust figure size
                ax1.plot(transformed_data.index, transformed_data[f'{country}_employment_value'], marker='o', color='orange')
                ax1.set_title(f'ICT Employment Data for {country}', fontsize=12)
                ax1.set_xlabel('Quarter', fontsize=10)
                ax1.set_ylabel('Percentage of Total Employees', fontsize=10)
                ax1.grid(True)  # Add grid to the plot
                ax1.tick_params(axis='x', rotation=45, labelsize=9)
                ax1.tick_params(axis='y', labelsize=9)
                st.pyplot(fig1)

                st.write("**Labour Demand Data**")
                fig3, ax3 = plt.subplots(figsize=(4, 2.5))  # Adjust figure size
                ax3.plot(transformed_data.index, transformed_data[f'{country}_labour_demand_value'], marker='o', color='orange')
                ax3.set_title(f'Labour Demand Data for {country}', fontsize=12)
                ax3.set_xlabel('Quarter', fontsize=10)
                ax3.set_ylabel('Percentage of Total Job Advertisements Online', fontsize=9)
                ax3.grid(True)  # Add grid to the plot
                ax3.tick_params(axis='x', rotation=45, labelsize=9)
                ax3.tick_params(axis='y', labelsize=9)
                st.pyplot(fig3)

                st.write("**GVA Data**")
                fig2, ax2 = plt.subplots(figsize=(4, 2.5))  # Adjust figure size
                ax2.plot(transformed_data.index, transformed_data[f'{country}_GVA_value'], marker='o', color='yellow')
                ax2.set_title(f'GVA Data for {country}', fontsize=12)
                ax2.set_xlabel('Quarter', fontsize=10)
                ax2.set_ylabel('Percentage of GDP', fontsize=10)
                ax2.grid(True)  # Add grid to the plot  
                ax2.tick_params(axis='x', rotation=45, labelsize=9)
                ax2.tick_params(axis='y', labelsize=9)
                st.pyplot(fig2)
                tab_idx += 1
            # Column 2 content: Index plot and bubble chart
             with col2:
                st.write(f"**DTPI Indicator for {country}**") 
                
                # set plot width
                plot_width = 800
                dpi_fig = 200

                fig_index, ax_index = plt.subplots(figsize=(plot_width/dpi_fig, 2.5), dpi = dpi_fig)  # Adjust figure size
                ax_index.plot(index_data.index, index_data[f'{country}'], marker='x', label=f'{country}', color='red')
                ax_index.set_title(f'Indicator for {country}', fontsize=12)
                ax_index.set_xlabel('Quarter', fontsize=10)
                ax_index.set_ylabel('Indicator Value', fontsize=10)
                ax_index.grid(True)  # Add grid to the plot
                ax_index.tick_params(axis='x', rotation=45, labelsize=9)
                ax_index.tick_params(axis='y', labelsize=9)
                st.pyplot(fig_index)
                
                def plot_heatmap_plotly(transformed_data, index_data, country):
                    # Prepare data for the heatmap (GVA, Employment, Labour Demand)
                    heatmap_data = transformed_data[[f'{country}_GVA_normalized_moving_average_value', 
                                                    f'{country}_employment_normalized_moving_average_value', 
                                                    f'{country}_labour_demand_normalized_moving_average_value']]
                    heatmap_data.columns = ['GVA', 'Employment', 'Labour Demand']
                    heatmap_data[' '] = np.nan  # nan column to create a space in the heatmap

                    # Add the index data as a new row to the heatmap
                    index_row = pd.DataFrame(index_data[f'{country}']).T
                    #index_row.index = ['Index']

                    # Combine the original heatmap data with the index data
                    combined_data = pd.concat([heatmap_data.T, index_row], axis=0)

                    # Plotly heatmap
                    fig = px.imshow(combined_data, 
                                    labels=dict(x="Quarter", y="Metric", color="Normalized Value"),
                                    x=heatmap_data.index,
                                    y=combined_data.index,
                                    color_continuous_scale='RdBu_r')
                    
                    fig.update_layout(title=f'Heatmap for {country} - GVA, Employment, Labour Demand, and Indicator',
                                    xaxis_nticks=36,
                                    width=plot_width + 100,  # Adjust width to match layout
                                    height=600,  # Adjust height to align with left column
                                    yaxis_title='Metric',    # Label for y-axis
                                    plot_bgcolor='#002f6c',  # Background color
                                    paper_bgcolor='#002f6c',  # Paper (outer plot) background color
                                    font=dict(color='#e5e5e5')  # Font color
                    )
                                    
                    st.plotly_chart(fig)
                plot_heatmap_plotly(transformed_data, index_data, f'{country}')
             
            #  st.markdown(f'---')
             st.markdown(f'### Historical Analysis and Highlights for {country_titles[idx]} DPTI Indicator')
             
             # TODO code to be refactored in a renderer function
             
             # In case of missing country, no rendering, but also no error
             try:
                highlights_per_year_quarter = description_text_by_quarter('EU' if 'EU' in country else country)
                debug_print(f'>>> Contents for  {country}')
                debug_print(json.dumps(highlights_per_year_quarter, indent=2))

                # Time to render the markdown contents, making visible always the last quarter from the last year
                collapsed = False
                years = sorted(list(highlights_per_year_quarter.keys()), reverse=True)
                # Going over the years, and the quarters in the year, it retrieves the contents and prepares for
                # formatting and visualisation, leveraging the markdown renderer
                for year in years:
                    quarters = sorted(list(highlights_per_year_quarter[year].keys()), reverse=True)
                    for quarter in quarters:
                        st.markdown('---', unsafe_allow_html=True, help=None)
                        if not collapsed:
                            st.markdown(f'<details open><summary>{year} {quarter}</summary>{highlights_per_year_quarter[year][quarter]}</details>', unsafe_allow_html=True, help=None)
                            collapsed = not collapsed
                            continue
                        st.markdown(f'<details><summary>{year} {quarter}</summary>{highlights_per_year_quarter[year][quarter]}</details>', unsafe_allow_html=True, help=None)
             except KeyError:
                 error_print(f'{country} data is not available: no rendering')
