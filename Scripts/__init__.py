import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


def custom(df, items):
    encoded_vals = []
    for index, row in df.iterrows():
        labels = {}
        uncommons = list(set(items) - set(row))
        commons = list(set(items).intersection(row))
        for uc in uncommons:
            labels[uc] = 0
        for com in commons:
            labels[com] = 1
        encoded_vals.append(labels)
    encoded_vals[0]

    ohe_df = pd.DataFrame(encoded_vals)

    freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
    freq_items.head(7)

    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
    rules.head()

    plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
    plt.xlabel('support')
    plt.ylabel('confidence')
    plt.title('Support vs Confidence')
    plt.show()


def main():
    url = 'https://gist.githubusercontent.com/Harsh-Git-Hub/2979ec48043928ad9033d8469928e751/raw/72de943e040b8bd0d087624b154d41b2ba9d9b60/retail_dataset.csv'
    df = pd.read_csv(url, sep=',')
    # print(df)

    items = (df['0'].unique())
    # print (items)

    custom(df, items)


main()