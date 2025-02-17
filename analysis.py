#!/usr/bin/env python3.12
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz


# Ukol 1: nacteni dat ze ZIP souboru
def load_data(filename: str, ds: str) -> pd.DataFrame:
    with zipfile.ZipFile(filename, "r") as zip_file:
        years = [file_info for file_info in zip_file.infolist() if file_info.is_dir()]

        final_df = pd.DataFrame()

        for year in years:
            target_file = f"{year.filename}I{ds}.xls"
            with zip_file.open(target_file) as file:
                df = pd.read_html(file, encoding="cp1250")
                final_df = pd.concat([df[0], final_df])

        return final_df.loc[:, ~final_df.columns.str.contains("^Unnamed")]


# Ukol 2: zpracovani dat
def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    dataframe = df.copy(deep=True)

    df = df.drop_duplicates(subset="p1")

    dataframe["date"] = pd.to_datetime(df["p2a"], dayfirst=True)

    regions = {
        0: "PHA",
        1: "STC",
        2: "JHC",
        3: "PLK",
        4: "ULK",
        5: "HKK",
        6: "JHM",
        7: "MSK",
        14: "OLK",
        15: "ZLK",
        16: "VYS",
        17: "PAK",
        18: "LBK",
        19: "KVK",
    }
    dataframe["region"] = df["p4a"].map(regions)

    if verbose:
        file_size = dataframe.memory_usage(deep=True).sum() / (10**6)
        print(f"new_size = {file_size:.1f} MB")

    return dataframe


# Ukol 3: počty nehod v jednotlivých regionech podle stavu vozovky
def plot_state(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    surface_type = {
        1: "povrch suchý",
        2: "povrch suchý",
        3: "povrch mokrý",
        4: "na vozovce je bláto",
        5: "na vozovce je náledí, ujetý sníh",
        6: "na vozovce je náledí, ujetý sníh",
    }

    filtered_df = df[df["p16"].isin(surface_type.keys())]

    accidents_df = pd.DataFrame()
    accidents_df["surface_type"] = filtered_df["p16"].map(surface_type)
    accidents_df["region"] = filtered_df["region"]

    accidents_count = (
        accidents_df.groupby(["surface_type", "region"])
        .size()
        .reset_index(name="count")
    )

    fig = sns.FacetGrid(
        accidents_count,
        col="surface_type",
        col_wrap=2,
        sharey=False,
        height=6,
        sharex=False,
    )

    # Map a barplot onto the grid
    fig.map(
        sns.barplot, "region", "count", order=sorted(accidents_df["region"].unique())
    )

    fig.set_titles(col_template="Stav povrchu vozovky: {col_name}")
    fig.set_axis_labels("Region", "Počet nehod")
    fig.tight_layout()

    plt.savefig(fig_location, bbox_inches="tight")
    if show_figure:
        plt.show()


# Ukol4: alkohol a následky v krajích
def plot_alcohol(
    df: pd.DataFrame,
    df_consequences: pd.DataFrame,
    fig_location: str = None,
    show_figure: bool = False,
):
    consequence_type = {
        1: "usmrcení",
        2: "těžké zranění",
        3: "lehké zranění",
        4: "bez zranění",
    }

    df = df[df["p11"] >= 3]
    df = df[df["p2b"].notna()]

    df_consequences["consequence_type"] = df_consequences["p59g"].map(consequence_type)
    merged_df = pd.merge(df, df_consequences, on="p1")

    merged_df["role"] = merged_df["p59a"].apply(
        lambda x: "řidič" if x == 1 else "pasažer"
    )

    # Create a DataFrame with all combinations so regions with zero deaths are also visualized
    all_combinations = pd.MultiIndex.from_product(
        [
            consequence_type.values(),
            merged_df["region"].unique(),
            ["řidič", "pasažer"],
        ],
        names=["consequence_type", "region", "role"],
    )

    accidents_count = (
        merged_df.groupby(["consequence_type", "region", "role"])
        .size()
        .reindex(all_combinations, fill_value=0)
        .reset_index(name="count")
    )

    fig = sns.FacetGrid(
        accidents_count,
        col="consequence_type",
        col_wrap=2,
        sharey=False,
        height=6,
        sharex=False,
    )

    fig.map_dataframe(
        sns.barplot, x="region", y="count", hue="role", hue_order=["řidič", "pasažer"]
    )

    fig.add_legend(title="Osoba")
    fig.set_axis_labels("Kraj", "Počet nehod")
    fig.set_titles(col_template="{col_name}")
    fig.tight_layout()

    plt.savefig(fig_location, bbox_inches="tight")
    if show_figure:
        plt.show()


# Ukol 5: Druh nehody (srážky) v čase
def plot_type(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    accident_type = {
        1: "srážka",
        2: "srážka s vozidlem zaparkovaným",
        3: "srážka s pevnou překážkou",
        4: "srážka s chodcem",
        5: "srážka s lesní zvěří",
        6: "srážka s domácím zvířetem",
        7: "srážka s vlakem",
        8: "srážka s tramvají",
        9: "havárie",
        0: "jiný druh nehody",
    }

    df = df[df["region"].isin(["JHM", "PHA", "JHC", "ULK"])]
    df["accident_type"] = df["p6"].map(accident_type)
    df["p2a"] = pd.to_datetime(df["p2a"], format="%d.%m.%Y")
    df["month"] = df["p2a"].dt.month

    pivot_table = pd.pivot_table(
        df,
        index=["p2a", "region"],
        columns="accident_type",
        values="p1",
        aggfunc="count",  # Count accidents
    )

    pivot_table_filtered = pivot_table.loc[
        ("2023-01-01" <= pivot_table.index.get_level_values("p2a"))
        & (pivot_table.index.get_level_values("p2a") <= "2024-10-01")
    ]

    # Group data by region and month
    pivot_table_monthly = pivot_table_filtered.groupby(
        ["region", pd.Grouper(level="p2a", freq="M")]
    ).sum()

    pivot_table_monthly = pivot_table_monthly.reset_index()
    regions = pivot_table_monthly["region"].unique()

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))

    # Itterate over regions to plot their data
    for i, region in enumerate(regions):
        ax = axs[i // 2, i % 2]  # Get correct position in the 2x2 grid
        region_data = pivot_table_monthly[pivot_table_monthly["region"] == region]

        lines = []
        labels = []

        # Plot each accident type
        for accident in region_data.columns[2:]:
            (line,) = ax.plot(region_data["p2a"], region_data[accident], linewidth=2)
            lines.append(line)
            labels.append(accident)

        ax.set_xlabel("Datum")
        ax.set_ylabel("Počet nehod")
        ax.set_title(f"Nehody v kraji {region}")
        ax.grid(True)
        ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-10-01"))

    fig.legend(lines, labels, title="Typ nehody", bbox_to_anchor=(1, 0.5), loc="center")

    plt.tight_layout()

    plt.savefig(fig_location, bbox_inches="tight")
    if show_figure:
        plt.show()


# %%
if __name__ == "__main__":
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni
    # funkce.

    df = load_data("data_23_24.zip", "nehody")
    df2 = parse_data(df, True)
    df_consequences = load_data("data_23_24.zip", "nasledky")

    plot_state(df2, "01_state.png", True)

    plot_alcohol(df2, df_consequences, "02_alcohol.png", True)

    plot_type(df2, "03_type.png", True)
