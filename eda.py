"""
Exploratory Data Analysis for the Counterfeit Medicine Dataset.
"""

import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import DATA_PATH, EDA_DIR


def save_plot(filename):
    """Save the current plot."""

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, filename))
    plt.close()


def plot_label_distribution(df):
    """Plot genuine vs counterfeit medicines."""

    counts = df["label"].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Counterfeit", "Genuine"],
        [counts.get(0, 0), counts.get(1, 0)]
    )

    plt.title("Medicine Distribution")
    plt.ylabel("Count")

    save_plot("label_distribution.png")


def plot_price_distribution(df):
    """Compare medicine prices."""

    plt.figure(figsize=(8, 4))

    plt.hist(
        df[df["label"] == 1]["price_usd"],
        bins=30,
        alpha=0.6,
        label="Genuine"
    )

    plt.hist(
        df[df["label"] == 0]["price_usd"],
        bins=30,
        alpha=0.6,
        label="Counterfeit"
    )

    plt.title("Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()

    save_plot("price_distribution.png")


def plot_top_manufacturers(df):
    """Show the top manufacturers."""

    top = df["manufacturer"].value_counts().head(10)

    plt.figure(figsize=(8, 5))

    plt.barh(top.index[::-1], top.values[::-1])

    plt.title("Top Manufacturers")

    save_plot("top_manufacturers.png")


def plot_country_distribution(df):
    """Show medicines by country."""

    counts = df["manufacture_country"].value_counts()

    plt.figure(figsize=(8, 4))

    plt.bar(counts.index, counts.values)

    plt.xticks(rotation=45)

    plt.title("Medicines by Country")

    save_plot("country_distribution.png")


def plot_country_risk(df):
    """Counterfeit percentage for each country."""

    risk = (
        df.groupby("manufacture_country")["label"]
        .apply(lambda value: (1 - value.mean()) * 100)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 4))

    plt.bar(risk.index, risk.values)

    plt.xticks(rotation=45)

    plt.ylabel("Counterfeit %")

    plt.title("Country Risk")

    save_plot("country_risk.png")


def main():

    print("Running EDA...\n")

    os.makedirs(EDA_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df["manufacture_date"] = pd.to_datetime(
        df["manufacture_date"],
        errors="coerce"
    )

    df["expiry_date"] = pd.to_datetime(
        df["expiry_date"],
        errors="coerce"
    )

    print(f"Dataset Size : {len(df)}")
    print(f"Total Columns : {len(df.columns)}")

    plot_label_distribution(df)
    plot_price_distribution(df)
    plot_top_manufacturers(df)
    plot_country_distribution(df)
    plot_country_risk(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
