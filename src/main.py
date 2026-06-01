# src/main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features.new_feature import new_feature
from src.engines.new_engine import new_engine

def main():
    # Example data
    data = {
        'x': [1, 2, 3, 4, 5],
        'y': [2, 3, 5, 7, 11]
    }
    df = pd.DataFrame(data)

    # Simple data analysis: Calculate mean and standard deviation
    mean_x = df['x'].mean()
    std_x = df['x'].std()

    print(f"Mean of x: {mean_x}")
    print(f"Standard Deviation of x: {std_x}")

    # Plotting
    plt.plot(df['x'], df['y'])
    plt.title('Simple Line Chart')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('plot.png')

    return df

if __name__ == '__main__':
    main()
    new_engine()