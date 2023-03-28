import pandas as pd


def cleaning_set(dataframe):
    dataframe = dataframe.drop(["slp","caa","thall"], axis=1)

    return dataframe


def main():
    try:
        dataframe = pd.read_csv("heart.csv")
        dataframe = cleaning_set(dataframe)
        dataframe.to_csv('heart2.csv', index=False)
        print(dataframe)
    except FileNotFoundError as e:
        print(e)
        print("file not found")


main()