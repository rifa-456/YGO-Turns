from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table

from training.turns import SPLITS, load_split

console = Console()


def match_summary(df: pd.DataFrame) -> Table:
    matches = df["match_id"].nunique()
    duels = df.groupby(["match_id", "duel_number"]).ngroups
    sizes = df.groupby(["match_id", "duel_number"]).size()

    table = Table(title="Dataset Overview", header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    rows = {
        "Total rows": f"{len(df):,}",
        "Total columns": f"{len(df.columns):,}",
        "Unique matches": f"{matches:,}",
        "Unique duels": f"{duels:,}",
        "Avg turns / duel": f"{len(df) / duels:.2f}",
        "Median turns / duel": f"{sizes.median():.1f}",
        "Max turns / duel": f"{sizes.max()}",
        "Avg duels / match": f"{duels / matches:.2f}",
        "P1 win rate": f"{df['p1_won'].mean() * 100:.1f}%",
        "P1 turn rate": f"{df['is_p1_turn'].mean() * 100:.1f}%",
    }
    for metric, value in rows.items():
        table.add_row(metric, value)
    return table


def zone_percentiles(df: pd.DataFrame) -> Table:
    cols = [
        "p1_hand_size",
        "p2_hand_size",
        "p1_deck_size",
        "p2_deck_size",
        "p1_gy_size",
        "p2_gy_size",
        "p1_banished_size",
        "p2_banished_size",
        "p1_extra_size",
        "p2_extra_size",
    ]
    quantiles = df[cols].quantile([0.5, 0.9, 0.95, 0.99])

    table = Table(title="Zone Size Percentiles", header_style="bold magenta")
    table.add_column("Column", style="dim")
    for label in ("p50", "p90", "p95", "p99", "max"):
        table.add_column(label, justify="right")

    for col in cols:
        s = df[col]
        table.add_row(
            col,
            f"{quantiles.loc[0.50, col]:.0f}",
            f"{quantiles.loc[0.90, col]:.0f}",
            f"{quantiles.loc[0.95, col]:.0f}",
            f"{quantiles.loc[0.99, col]:.0f}",
            f"{s.max():.0f}",
        )
    return table


def occupancy_summary(df: pd.DataFrame) -> Table:
    table = Table(title="Zone Occupancy", header_style="bold magenta")
    table.add_column("Player", style="dim")
    table.add_column("Zone", style="dim")
    table.add_column("Avg Occupied", justify="right")
    table.add_column("Max", justify="right")

    for zone, n_slots in [("monster_zone", 5), ("spell_zone", 5)]:
        for player in ["p1", "p2"]:
            cols = [f"{player}_{zone}_{i}_name" for i in range(n_slots)]
            occupied = df[cols].apply(lambda c: (c != "<PAD>") & c.notna()).sum(axis=1)
            table.add_row(player, zone, f"{occupied.mean():.2f}", f"{occupied.max()}")
    return table


def top_hand_cards(df: pd.DataFrame, n: int = 20) -> Table:
    hand_cols = [f"{p}_hand_{i}" for p in ["p1", "p2"] for i in range(10)]
    cards = df[hand_cols].values.flatten()
    s = pd.Series(cards)
    counts = s[s.notna() & (s != "<PAD>")].value_counts().head(n)

    table = Table(title=f"Top {n} Hand Cards", header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Card")
    table.add_column("Count", justify="right")

    for rank, (card, count) in enumerate(counts.items(), start=1):
        table.add_row(str(rank), card, f"{count:,}")
    return table


def main() -> None:
    df = pd.concat([load_split(split) for split in SPLITS], ignore_index=True)
    for table in (match_summary, zone_percentiles, occupancy_summary, top_hand_cards):
        console.print(table(df))
        console.print()


if __name__ == "__main__":
    main()
