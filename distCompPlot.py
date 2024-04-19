import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv('1000perData.csv')

# Plotting the distribution of MFCC_1 for each country
plt.figure(figsize=(14, 7))
sns.boxplot(data=data, x='Filename', y='MFCC_1')
plt.title('Distribution of MFCC_1 Across Countries')
plt.xlabel('Country')
plt.ylabel('MFCC_1 Value')
plt.xticks(rotation=45)
plt.show()
