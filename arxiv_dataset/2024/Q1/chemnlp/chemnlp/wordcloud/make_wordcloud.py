"""Module to generate word-cloud."""

import os
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud
import pandas as pd


def make_plot(df=None, key="title"):
    """Generate a word-cloud given pandas dataframe."""
    all_text = " ".join(df[key].values)

    f = open("sample.txt", "w")
    f.write(all_text)
    f.close()
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

    # Read the whole text.
    text = open(path.join(d, "sample.txt")).read()

    # Generate a word cloud image
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40, background_color="white").generate(
        text
    )
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("wordc.png")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("cond_mat.csv")

    key = "title"
    make_plot(df=df, key=key)
