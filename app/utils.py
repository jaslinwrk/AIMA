import pandas as pd

def clean_sales_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'Date': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

def merge_data(sales_df, weather_df):
    # Ensure date columns are in datetime format
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    merged_df = sales_df.merge(weather_df, on='date', how='left')
    
    return merged_df

def encode_categorical_columns(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category').cat.codes
    return df

def get_column_names(df):
    return df.columns.tolist()