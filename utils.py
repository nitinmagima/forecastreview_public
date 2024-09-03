# --------------------------------------------------------------------------------------------------
# Functions for Trigger Selection Analysis
#
# Author - Nitin Magima
# Date - 2024
# Version - 1.0
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
#
# IMPORTANT - DISCLAIMER AND RIGHTS STATEMENT
# This is a set of scripts written by the Financial Instruments Team at the International Research
# Institute for Climate and Society (IRI) part of The Columbia Climate School, Columbia University
# They are shared for educational purposes only.  Anyone who uses this code or its
# functionality or structure, assumes full liability and should inform and credit IRI.
#
# ==================================================================================================

# Loading Packages
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML
import yaml
import numpy as np
import matplotlib.colors as mcolors

def load_config(file_path="config.yaml"):
    """
    Loads configuration data from a YAML file.

    Args:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - dict: Dictionary containing configuration data.
    """    
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_data(maproom, mode, region, season, predictor, predictand, year, bad_years,
             issue_month0, freq, include_upcoming, threshold_protocol, username, password):
    """
    Retrieves data from an API endpoint and combines it into a DataFrame.
    ...
    """
    
    # Make a GET request to the API
    region_str = ",".join(map(str, region))  # Convert region values to a comma-separated string
    api_url = (f"https://iridl.ldeo.columbia.edu/fbfmaproom2/{maproom}/"
               f"export?season={season}&issue_month0={issue_month0}&freq={freq}&predictor"
               f"={predictor}&predictand={predictand}&include_upcoming={include_upcoming}&mode={mode}"
               f"&region={region_str}")

    # Constructing the design tool URL
    tool_url = (f"https://iridl.ldeo.columbia.edu/fbfmaproom2/{maproom}?mode={mode}&map_column={predictor}"
               f"&season={season}&predictors={predictor}&predictand={predictand}&year={year}"
               f"&issue_month0={issue_month0}&freq={freq}&severity=0&include_upcoming={include_upcoming}")

    response = requests.get(api_url, auth=(username, password))

    if response.status_code == 200:
        json_data = response.json()
        flattened_data = pd.json_normalize(json_data)

        non_nested_columns = []

        for column in flattened_data.columns:
            if isinstance(flattened_data[column][0], list):
                expanded_data = pd.json_normalize(flattened_data[column].explode(), sep='_')
                flattened_data = pd.concat([flattened_data, expanded_data], axis=1)
                flattened_data = flattened_data.drop(column, axis=1)
            else:
                non_nested_columns.append(column)

        non_nested_df = flattened_data[non_nested_columns]
        melted_non_nested_df = pd.DataFrame({
            'Metric': non_nested_df.columns,
            'Value': non_nested_df.iloc[0].values
        })

        replace_values = {
            'threshold': 'Forecast Threshold',
            'skill.accuracy': 'Forecast Accuracy',
            'skill.act_in_vain': 'Act in Vain',
            'skill.fail_to_act': 'Fail to Act',
            'skill.worthy_action': 'Worthy Action',
            'skill.worthy_inaction': 'Worthy Inaction'
        }
        melted_non_nested_df['Metric'] = melted_non_nested_df['Metric'].replace(replace_values)


        # Convert melted_non_nested_df to a dictionary
        melted_non_nested_dict = melted_non_nested_df.set_index('Metric')['Value'].to_dict()

        # Convert flattened data to Pandas DataFrame
        df = pd.DataFrame(flattened_data).drop(non_nested_df.columns, axis=1, errors='ignore')
        df['Triggered'] = df[predictor] > melted_non_nested_dict['Forecast Threshold']
        df['Trigger Difference'] = df[predictor] - melted_non_nested_dict['Forecast Threshold']
        df['Adjusted Forecast Threshold'] = melted_non_nested_dict['Forecast Threshold'] + threshold_protocol
        df['Triggered Adjusted'] = df[predictor] > melted_non_nested_dict['Forecast Threshold']
        df.rename(columns={predictor: 'Forecast', 'year': 'Year'}, inplace=True)

        # Filter df based on the provided list of years
        df = df[df['Year'].isin(bad_years)]
        
        # Select relevant columns including 'year', and no longer limiting the DataFrame to the first row
        df = df.loc[:, ['Year', 'Forecast', 'Trigger Difference', 'Triggered', 'Triggered Adjusted', 'Adjusted Forecast Threshold']]

        # Combine df and melted_non_nested_df
     
        combined_df = df

        melted_non_nested_df.set_index('Metric', inplace=True)

        combined_df['Act in Vain'] = melted_non_nested_df.at['Act in Vain', 'Value']
        combined_df['Fail to Act'] = melted_non_nested_df.at['Fail to Act', 'Value']
        combined_df['Worthy Action'] = melted_non_nested_df.at['Worthy Action', 'Value']
        combined_df['Worthy Inaction'] = melted_non_nested_df.at['Worthy Inaction', 'Value']
        combined_df['Frequency (%)'] = f"{freq}%"
        combined_df['Forecast Accuracy (%)'] = melted_non_nested_df.at['Forecast Accuracy', 'Value']
        combined_df['Forecast Threshold'] = melted_non_nested_df.at['Forecast Threshold', 'Value']
        combined_df['Threshold Protocol'] = f"{threshold_protocol}"

        month_mapping = {
            0: 'Jan',
            1: 'Feb',
            2: 'Mar',
            3: 'Apr',
            4: 'May',
            5: 'Jun',
            6: 'Jul',
            7: 'Aug',
            8: 'Sep',
            9: 'Oct',
            10: 'Nov',
            11: 'Dec'
        }

        combined_df['Issue Month'] = issue_month0
        combined_df['Issue Month'] = combined_df['Issue Month'].map(month_mapping)
        combined_df['Design Tool URL'] = f"<a href='{tool_url}'>Design Tool Link</a>"

        # Define the sequence of desired columns
        desired_columns = ['Year', 'Frequency (%)', 'Issue Month', 'Forecast', 'Forecast Threshold', 'Trigger Difference',
                           'Forecast Accuracy (%)', 'Triggered', 'Adjusted Forecast Threshold', 'Threshold Protocol',
                           'Triggered Adjusted', 'Act in Vain', 'Fail to Act', 'Worthy Action', 'Worthy Inaction',
                           'Design Tool URL']
        combined_df = combined_df.reindex(columns=desired_columns)

        combined_df = combined_df.rename(columns={
            'forecast': 'Forecast',
            'triggered': 'Triggered',
            'trigger difference': 'Trigger Difference'
            # Additional renaming handled by replace_values
        })

        return combined_df
    else:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()



def get_admin_data(maproom, level, username, password, need_valid_keys, valid_keys=None):
    """
    Retrieves administrative data from an API endpoint.

    Args:
    - maproom (str): Maproom value.
    - level (str): Level of administrative data.

    Returns:
    - DataFrame: DataFrame containing administrative data.
    """
    # Construct the API URL with the provided parameters
    api_url = f"https://iridl.ldeo.columbia.edu/fbfmaproom2/regions?country={maproom}&level={level}"

    # Make a GET request to the API
    if username and password:
        auth = (username, password)
        response = requests.get(api_url, auth=auth)
    else:
        response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON data
        json_data = response.json()

        # Create a DataFrame from the JSON data
        df = pd.DataFrame(json_data)

        # Extract "key" and "label" from the "regions" column
        df[['key', 'label']] = df['regions'].apply(pd.Series)

        # Filter keys if valid_keys is provided
        if level != 0:
            if need_valid_keys is True:
                df = df[df['key'].isin(valid_keys)]

        # Drop the original "regions" column if needed
        df = df.drop('regions', axis=1)

        return df
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        return None


def get_trigger_tables(maproom, mode, season, predictor, predictand, year, bad_years,
                       issue_month, frequencies, include_upcoming, threshold_protocol, username, password,
                       need_valid_keys, valid_keys):
    """
    Retrieves trigger tables based on specified parameters.

    Args:
    - maproom (str): Maproom value.
    - mode (int): Mode value.
    - season (str): Season value.
    - predictor (str): Predictor value.
    - predictand (str): Predictand value.
    - issue_month (list): List of issue month values.
    - frequencies (list): List of frequency values.
    - include_upcoming (str): Include upcoming value.
    - threshold_protocol (int): Threshold protocol value.
    - username (str): Username for API authentication.
    - password (str): Password for API authentication.
    - need_valid_keys (bool): Flag indicating if valid keys are needed.
    - valid_keys (list): List of valid keys.

    Returns:
    - dict: Dictionary containing trigger tables.
    """
    
    # Initialize a dictionary to store admin tables
    admin_tables = {}

    # Creating trigger tables

    admin_name = f"admin{mode}_tables"
    admin_tables[admin_name] = {}
    admin_data = get_admin_data(maproom, mode, username=username, password=password,
                                need_valid_keys=need_valid_keys, valid_keys=valid_keys)

    for freq in frequencies:
        for month in issue_month:
            # Iterate over each key value
            if isinstance(admin_data, pd.Series):
                for region_key, label in admin_data.items():
                    print(region_key, label)
                    table_name = f"output_freq_{freq}_mode_{mode}_month_{month}_region_{region_key}_table"

                    df = get_data(maproom=maproom, mode=mode, region=[region_key],
                                  season=season, predictor=predictor, predictand=predictand, year = year,
                                  issue_month0=month, freq=freq, include_upcoming=include_upcoming,
                                  bad_years = bad_years,
                                  threshold_protocol=threshold_protocol, username=username, password=password)

                    df.insert(0, 'Admin Name', label)
                    admin_tables[admin_name][table_name] = df

            elif isinstance(admin_data, pd.DataFrame):
                for index, row in admin_data.iterrows():
                    region_key, label = row['key'], row['label']

                    table_name = f"output_freq_{freq}_mode_{mode}_month_{month}_region_{region_key}_table"

                    df = get_data(maproom=maproom, mode=mode, region=[region_key],
                                  season=season, predictor=predictor, predictand=predictand, year = year,
                                  issue_month0=month, freq=freq, include_upcoming=include_upcoming,
                                  bad_years = bad_years,
                                  threshold_protocol=threshold_protocol, username=username, password=password)

                    df.insert(0, 'Admin Name', label)
                    admin_tables[admin_name][table_name] = df

            else:
                # Handle other cases or raise an error
                raise ValueError("Unexpected output type from get_admin_data.")

    return admin_tables

def generate_colors(n):
    """
    Generates a list of n distinct colors in HSL format.
    
    Args:
        n (int): The number of distinct colors to generate.
        
    Returns:
        List[str]: A list of colors.
    """
    return [f"hsl({int((360 / n) * i)}, 100%, 70%)" for i in range(n)]

def style_and_render_df_with_hyperlinks(df):
    # Define columns to style
    columns_to_style = ['Admin Name', 'Frequency (%)', 'Issue Month']
    
    # Calculate the total number of unique values across all columns
    unique_values_count = sum(df[col].nunique() for col in columns_to_style)
    
    # Generate a unique color for each unique value across all columns
    unique_colors = generate_colors(unique_values_count)
    
    # Assign a distinct segment of colors to each column
    color_index = 0
    color_maps = {}
    for column in columns_to_style:
        unique_values = df[column].unique()
        color_maps[column] = {value: unique_colors[color_index + i] for i, value in enumerate(unique_values)}
        color_index += len(unique_values)
    
    # Function to apply colors based on the value for a given column
    def apply_color(val, column):
        if pd.isnull(val):
            return ''  # Return default style for NaN values
        return f'background-color: {color_maps[column].get(val, "")};'
    
    # Initialize the styled DataFrame
    styled_df = df.style
    
    # Apply the styles to each column individually
    for column in columns_to_style:
        styled_df = styled_df.map(lambda val, col=column: apply_color(val, col), subset=[column])
    
    # Apply boolean highlights for 'triggered' and 'Triggered Adjusted' columns
    true_color, false_color = '#CCFFCC', '#FFCC99'
    
    # Assuming styled_df is your DataFrame styled object, and true_color/false_color are defined
    columns_to_style = ['Triggered', 'Triggered Adjusted']
    
    for col in columns_to_style:
        try:
            # Check if column exists by trying to access it
            if col in styled_df.columns:
                # Apply the styling if column exists
                styled_df = styled_df.map(lambda val: f'background-color: {true_color if val else false_color}', subset=[col])
            else:
                # If the column does not exist in the DataFrame, this line will not be executed
                pass
        except KeyError as e:
            print(f"Column not found: {e}")
            # Handle the case where the column does not exist, e.g., by logging or passing
            continue
    
    # Format numerical columns
    styled_df = styled_df.format({'Forecast': "{:.2f}", 'Trigger Difference': "{:.2f}", 'Forecast Accuracy (%)': "{:.2%}",'Forecast Threshold': "{:.2f}", 'Act in Vain': "{:.1f}", 'Fail to Act': "{:.1f}", 'Worthy Action': "{:.1f}", 'Worthy Inaction': "{:.1f}", 'Adjusted Forecast Threshold': "{:.2f}" })
    
    # Render to HTML
    rendered_html = styled_df.to_html(escape=False)
    display(HTML(rendered_html))

def plot_triggered_events_admin0(data, season, severity):
    """
    Plots count of triggered events by year with 'Issue Month' and 'Frequency (%)' as hues.

    Parameters:
    - data: DataFrame containing 'Year', 'Issue Month', and 'Frequency (%)' columns.
    - season (string): Target Season
    - severity (string): Severity Level
    """
    
    # Group the data by the 'Year' and count the number of triggered events
    yearly_triggered_count = data.groupby(['Year']).size().reset_index(name='Count of Triggered')
    
    # Set up a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Count plot with 'Issue Month' as the hue
    sns.countplot(data=data, x='Year', hue='Issue Month', ax=axes[0], dodge=True)
    axes[0].set_title(f'Triggered Events Frequency by Year and Issue Month - {season} {severity}')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Count of Triggered Events')
    
    # Count plot with 'Frequency (%)' as the hue
    sns.countplot(data=data, x='Year', hue='Frequency (%)', ax=axes[1], dodge=True)
    axes[1].set_title(f'Triggered Events Frequency by Year and Frequency (%) - {season} {severity}')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Count of Triggered Events')
    
    plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
    plt.show()

def plot_triggered_events_heatmap_admin0(data, season, severity):
    """
    Plots a heatmap of triggered events summarized by frequency and issue month.

    Parameters:
    - data: DataFrame containing 'Frequency (%)', 'Issue Month', and count of triggered events.
    - season (string): Target Season
    - severity (string): Severity Level
    """
    # Summarize the counts by frequency and issue month
    triggered_by_freq_month = data.groupby(['Frequency (%)', 'Issue Month']).size().unstack(fill_value=0)
    
    # Creating the heatmap
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(triggered_by_freq_month, annot=True, cmap="YlOrRd", fmt="d")
    plt.title(f'Heatmap of Triggered Events - {season} {severity}')
    plt.xlabel('Issue Month')
    plt.ylabel('Frequency (%)')
    
    plt.show()
    plt.clf()  # Clear the figure to avoid overlap in subsequent plots

def plot_trigger_difference_boxplot_admin0(data, season, severity):
    """
    Plots a boxplot for the distribution of values in a specified column, including outliers and quartiles.

    Parameters:
    - data: DataFrame containing the data to plot.
    - season (string): Target Season
    - severity (string): Severity Level
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data['Trigger Difference'])
    plt.title(f'Trigger Difference Boxplot - {season} {severity}')
    plt.xlabel('Trigger Difference')
    plt.grid(True)
    plt.show
    # Calculate and display the quantile ranges for Trigger Difference values for the current Admin Name
    quantiles_admin = data['Trigger Difference'].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
    print(f"Quantile Ranges for Trigger Difference - {season} {severity} {quantiles_admin}")
    print(quantiles_admin)
    print("\n")

def generate_heatmaps_for_admin1(data, season, severity):
    """
    Generates and optionally saves heatmaps for triggered events summarized by frequency and issue month for each administrative name.

    Parameters:
    - data: DataFrame containing the data.
    - season (string): Target Season
    - severity (string): Severity Level
    """
    # Filter for triggered events
    triggered_events = data[data['Triggered']]

    # Get unique admin names
    admin_names = triggered_events['Admin Name'].unique()

    for admin in admin_names:
        # Filter the data for the current Admin Name
        admin_triggered_events = triggered_events[triggered_events['Admin Name'] == admin]

        # Summarize the counts by frequency and issue month
        admin_triggered_by_freq_month = admin_triggered_events.groupby(['Frequency (%)', 'Issue Month']).size().unstack(fill_value=0)

        # Generate and display the heatmap
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(admin_triggered_by_freq_month, annot=True, cmap="BuPu", fmt="d")
        title = f'Heatmap of Triggered Events for {admin} - {season} {severity}'
        plt.title(title)
        plt.xlabel('Issue Month')
        plt.ylabel('Frequency (%)')
        plt.show()
        plt.clf()  # Clear the figure for the next iteration

def plot_boxplots_and_quantiles_admin1(data, season, severity):
    """
    Generates boxplots and displays quantile ranges for the Trigger Difference column for each value in the Admin column.

    Parameters:
    - data: DataFrame containing the data.
    - season (string): Target Season
    - severity (string): Severity Level
    """
    # Looping through each unique Admin Name
    admin_names = data['Admin Name'].unique()
    
    for admin in admin_names:
        # Filter the data for the current Admin Name
        admin_data = data[data['Admin Name'] == admin]
        
        # Boxplot for Trigger Difference values for the current Admin Name
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=admin_data['Trigger Difference'])
        plt.title(f'Boxplot of Trigger Difference Values for {admin} - {season} {severity}')
        plt.xlabel('Trigger Difference')
        plt.grid(True)
        plt.show()
        
        # Calculate and display the quantile ranges for Trigger Difference values for the current Admin Name
        quantiles_admin = admin_data['Trigger Difference'].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict()
        print(f"Quantile Ranges for {admin}:")
        print(quantiles_admin)
        print("\n")

def visualize_decision_outcomes(grouped_data):
    """
    Visualizes decision outcomes for each frequency across different administrative names
    and issue months using a 2x2 grid of bar plots. Each subplot represents one of the outcomes:
    'Worthy Action', 'Act in Vain', 'Worthy Inaction', 'Fail to Act'.
    
    Parameters:
    - grouped_data: DataFrame containing the aggregated data to be visualized. 
      It must include columns for 'Frequency (%)', 'Admin Name', 'Issue Month', and the outcomes 
      ('Worthy Action', 'Act in Vain', 'Worthy Inaction', 'Fail to Act').
    
    Returns:
    None. Displays a matplotlib figure with the plots.
    """
    # Define the outcomes to visualize
    outcomes = ['Worthy Action', 'Act in Vain', 'Worthy Inaction', 'Fail to Act']

    # Get unique issue months from the DataFrame
    issue_months = grouped_data['Issue Month'].unique()

    for month in issue_months:
        # Filter data for the current issue month
        month_data = grouped_data[grouped_data['Issue Month'] == month]

        # Set up a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(10,10))  # Adjust the figsize as needed
        axes = axes.flatten()  # Flatten the 2D array of axes to iterate over them

        for i, outcome in enumerate(outcomes):
            ax = axes[i]
            sns.barplot(ax=ax, data=month_data, x='Frequency (%)', y=outcome, hue='Admin Name', errorbar=None)
            ax.set_title(f'{outcome} by Frequency (%) and Admin Name in {month}')
            ax.set_ylabel('Average Outcome')
            ax.set_xlabel('Frequency (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Admin Name', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


def plot_metrics_by_admin_and_frequency(grouped_data):
    """
    For each unique 'Admin Name' and 'Issue Month' in the DataFrame, calculates and plots 
    various metrics based on 'Worthy Action', 'Act in Vain', 'Worthy Inaction', and 'Fail to Act'
    values against frequency.

    Parameters:
    - grouped_data: A pandas DataFrame expected to contain columns 'Admin Name', 'Issue Month',
      'Frequency (%)', 'Worthy Action', 'Act in Vain', 'Worthy Inaction', and 'Fail to Act'.

    Returns:
    None. Displays a matplotlib graph for each 'Admin Name' and 'Issue Month'.
    """
    # Replace infinite values with NaN
    grouped_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values to avoid plotting issues
    grouped_data.dropna(subset=['Hit Rate (HR)', 'False Alarm Ratio (FAR)', 'Bias Score (BS)',
                                'Hanssen-Kuipers Score (KSS)', 'Heidke Skill Score (HSS)'], inplace=True)

    # Get unique admin names and issue months from the DataFrame
    admin_names = grouped_data['Admin Name'].unique()
    issue_months = grouped_data['Issue Month'].unique()

    for admin in admin_names:
        for month in issue_months:
            # Filter data for the current admin and issue month
            admin_month_data = grouped_data[(grouped_data['Admin Name'] == admin) & 
                                            (grouped_data['Issue Month'] == month)]

            # Plot the metrics for the current admin and issue month
            plt.figure(figsize=(8, 5))
            sns.lineplot(x='Frequency (%)', y='Hit Rate (HR)', data=admin_month_data, marker='o', label='Hit Rate (HR)')
            sns.lineplot(x='Frequency (%)', y='False Alarm Ratio (FAR)', data=admin_month_data, marker='s', label='False Alarm Ratio (FAR)')
            sns.lineplot(x='Frequency (%)', y='Bias Score (BS)', data=admin_month_data, marker='^', label='Bias Score (BS)')
            sns.lineplot(x='Frequency (%)', y='Hanssen-Kuipers Score (KSS)', data=admin_month_data, marker='^', label='Hanssen-Kuipers Score (KSS)')
            sns.lineplot(x='Frequency (%)', y='Heidke Skill Score (HSS)', data=admin_month_data, marker='^', label='Heidke Skill Score (HSS)')

            plt.title(f'Metrics vs Frequency for {admin} in {month}')
            plt.ylabel('Metric Value')
            plt.xlabel('Frequency (%)')
            plt.legend()
            plt.tight_layout()
            plt.show()


def calculate_forecast_metrics(dataframe):
    # Replace infinite values with NaN
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Get unique admin names from the DataFrame
    admin_names = dataframe['Admin Name'].unique()

    results = []  # List to store results for each admin

    for admin in admin_names:
        # Filter data for the current admin
        admin_data = dataframe[dataframe['Admin Name'] == admin].copy()

        # Calculate metrics for the filtered admin data

        # Calculating Hit Rate (HR)
        admin_data['Hit Rate (HR)'] = admin_data['Worthy Action'] / (admin_data['Worthy Action'] + admin_data['Fail to Act'])

        # Calculating False Alarm Ratio (FAR)
        admin_data['False Alarm Ratio (FAR)'] = admin_data['Act in Vain'] / (admin_data['Worthy Action'] + admin_data['Act in Vain'])

        # Calculating Bias Score (BS)
        admin_data['Bias Score (BS)'] = (admin_data['Worthy Action'] + admin_data['Act in Vain']) / \
                                        (admin_data['Worthy Action'] + admin_data['Fail to Act'])

        # Calculating Hanssen-Kuipers Score (KSS)
        admin_data['Hanssen-Kuipers Score (KSS)'] = admin_data['Hit Rate (HR)'] - admin_data['False Alarm Ratio (FAR)']

        # Calculating Heidke Skill Score (HSS)
        admin_data['Heidke Skill Score (HSS)'] = (2 * (admin_data['Worthy Action'] * admin_data['Worthy Inaction'] -
                                                       admin_data['Act in Vain'] * admin_data['Fail to Act'])) / \
                                                  ((admin_data['Worthy Action'] + admin_data['Fail to Act']) *
                                                   (admin_data['Fail to Act'] + admin_data['Worthy Inaction']) +
                                                   (admin_data['Worthy Action'] + admin_data['Act in Vain']) *
                                                   (admin_data['Act in Vain'] + admin_data['Worthy Inaction']))


        # Collecting processed data for each admin
        results.append(admin_data)

    # Combine all results into a single DataFrame
    final_results = pd.concat(results, ignore_index=True)

    return final_results

def format_forecast_metrics(dataframe):
    # Apply formatting
    formatted_final_results = dataframe.style.format({
        'Hit Rate (HR)': "{:.2f}",
        'False Alarm Ratio (FAR)': "{:.2f}",
        'Bias Score (BS)': "{:.2f}",
        'Hanssen-Kuipers Score (KSS)': "{:.2f}",
        'Heidke Skill Score (HSS)': "{:.2f}",
        'Act in Vain': "{:.1f}",
        'Fail to Act': "{:.1f}",
        'Worthy Action': "{:.1f}",
        'Worthy Inaction': "{:.1f}"
    })

    return formatted_final_results

def classify_trigger_HR_vs_FAR(df, threshold=0.6):
    """
    Classify each row in the DataFrame based on the hit rate and false alarm ratio into categories 'Good', 'Moderate', or 'Bad'.
    
    This function adds a 'Classification' column to the DataFrame where the classification is determined by
    comparing 'Hit Rate (HR)' and 'False Alarm Ratio (FAR)' with each other and a specified threshold.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing 'Hit Rate (HR)' and 'False Alarm Ratio (FAR)' columns.
    - threshold (float, optional): The threshold value for determining the classification. Defaults to 0.6.
    
    Returns:
    - pd.DataFrame: The original DataFrame with an additional 'Classification' column containing the classifications
                    ('Good', 'Moderate', 'Bad') based on the specified logic.
    """
    def classify_row(row):
        if (row['Hit Rate (HR)'] > row['False Alarm Ratio (FAR)']) and (row['Hit Rate (HR)'] > threshold):
            return 'Good'
        elif (row['Hit Rate (HR)'] >= row['False Alarm Ratio (FAR)']) and (row['Hit Rate (HR)'] < threshold):
            return 'Moderate'
        else:
            return 'Bad'
    
    # Apply the classification function to each row
    df['Classification'] = df.apply(classify_row, axis=1)
    return df

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_HR_FAR_classification_heatmap(data, index='Frequency (%)', columns='Issue Month', values='Classification', figsize=(8, 6)):
    """
    Plots a heatmap for the most common classification for each group of data using a color-coded table.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data to classify and visualize.
    - index (str): The column name to use as the rows of the pivot table.
    - columns (str): The column name to use as the columns of the pivot table.
    - values (str): The column name containing the classification values to visualize.
    - classification_map (dict): A dictionary mapping classification labels to colors.
    - figsize (tuple, optional): The size of the figure. Defaults to (8, 6).

    Returns:
    - None: The function displays the heatmap plot.
    """
    
    classification_map = {'Good': 'green', 'Moderate': 'yellow', 'Bad': 'red'}
    
    # Pivot the data to find the most common classification for each combination of index and columns
    classification_counts = data.pivot_table(index=index, columns=columns, values=values,
                                             aggfunc=lambda x: pd.Series.mode(x)[0])

    # Map the classifications to colors
    colored_matrix = classification_counts.applymap(lambda x: classification_map.get(x, 'white'))

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Create the table with colored cells
    the_table = ax.table(cellText=classification_counts.values,
                         rowLabels=classification_counts.index,
                         colLabels=classification_counts.columns,
                         cellColours=colored_matrix.applymap(lambda x: mcolors.to_rgba(x) if pd.notnull(x) else mcolors.to_rgba('white')).values,
                         cellLoc='center', loc='center')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 1.2)

    plt.title('Classification Heatmap Based on Hit Rate (HR) and False Alarm Ratio (FAR)')
    plt.show()
