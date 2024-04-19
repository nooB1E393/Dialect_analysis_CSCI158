import pandas as pd

# Load your data
data = pd.read_csv('1000perData.csv')


# Define a function to remove outliers based on IQR for all specified columns
def remove_outliers_all_columns(df, columns):
    clean_data = df.copy()  # Make a copy of the dataframe to retain the original data
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers in each column
        clean_data = clean_data[(clean_data[column] >= lower_bound) & (clean_data[column] <= upper_bound)]
    return clean_data


# List of MFCC columns to clean
mfcc_columns = ['MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7',
                'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13', 'MFCC_14', 'MFCC_15']

# Apply the function to clean all specified MFCC columns
cleaned_data = remove_outliers_all_columns(data, mfcc_columns)

# Display some of the cleaned data to verify
print(cleaned_data.head())

# Overwrite the original CSV file with the cleaned data
cleaned_data.to_csv('1000perData.csv', index=False)
