import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sqlite3

    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    return mo, mpl, pl, plt, sqlite3


@app.cell
def _(expand_json_column, pl, sqlite3):
    conn = sqlite3.connect("results/results.sqlite")

    df = pl.read_database("SELECT * FROM results", conn)
    df = expand_json_column(df, "exp_cfg")
    df
    return conn, df


@app.cell
def _(ALL_RGB01, df, pl, plt):
    tasks = sorted(df.get_column("task_name").unique().to_list())
    fig, axes = plt.subplots(
        ncols=len(tasks), squeeze=False, figsize=(6 * len(tasks), 5)
    )

    # Plot performance for each MLLM with respect to number of training samples.
    for task, ax in zip(tasks, axes[0]):
        for model, color in zip(
            sorted(df.get_column("model_ckpt").unique().to_list()),
            ALL_RGB01[1::2],
        ):
            for prompting in ("single", "multi"):
                filtered_df = df.filter(
                    (pl.col("model_ckpt") == model)
                    & (pl.col("task_name") == task)
                    & (pl.col("prompting") == prompting)
                ).sort("n_train")

                lowers = filtered_df.get_column("confidence_lower").to_list()
                means = filtered_df.get_column("mean_score").to_list()
                uppers = filtered_df.get_column("confidence_upper").to_list()
                xs = filtered_df.get_column("n_train").to_list()

                linestyle = "--" if prompting == "multi" else "-"

                ax.plot(
                    xs,
                    means,
                    marker="o",
                    label=f"Mean Score ({model.removeprefix('openrouter/')}, {prompting})",
                    color=color,
                    linestyle=linestyle,
                )
                ax.fill_between(xs, lowers, uppers, alpha=0.2, color=color, linewidth=0)
                ax.set_xlabel("Number of Training Samples")
                ax.set_ylabel("Score")
                ax.set_ylim(0, 1.0)
                ax.set_title(f"{task}")
                ax.set_xscale("symlog", linthresh=3)

    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()
    return (
        ax,
        axes,
        color,
        fig,
        filtered_df,
        linestyle,
        lowers,
        means,
        model,
        prompting,
        task,
        tasks,
        uppers,
        xs,
    )


@app.cell
def _():
    # https://coolors.co/palette/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226

    BLACK_HEX = "001219"
    BLACK_RGB = (0, 18, 25)
    BLACK_RGB01 = tuple(c / 256 for c in BLACK_RGB)

    BLUE_HEX = "005f73"
    BLUE_RGB = (0, 95, 115)
    BLUE_RGB01 = tuple(c / 256 for c in BLUE_RGB)

    CYAN_HEX = "0a9396"
    CYAN_RGB = (10, 147, 150)
    CYAN_RGB01 = tuple(c / 256 for c in CYAN_RGB)

    SEA_HEX = "94d2bd"
    SEA_RGB = (148, 210, 189)
    SEA_RGB01 = tuple(c / 256 for c in SEA_RGB)

    CREAM_HEX = "e9d8a6"
    CREAM_RGB = (233, 216, 166)
    CREAM_RGB01 = tuple(c / 256 for c in CREAM_RGB)

    GOLD_HEX = "ee9b00"
    GOLD_RGB = (238, 155, 0)
    GOLD_RGB01 = tuple(c / 256 for c in GOLD_RGB)

    ORANGE_HEX = "ca6702"
    ORANGE_RGB = (202, 103, 2)
    ORANGE_RGB01 = tuple(c / 256 for c in ORANGE_RGB)

    RUST_HEX = "bb3e03"
    RUST_RGB = (187, 62, 3)
    RUST_RGB01 = tuple(c / 256 for c in RUST_RGB)

    SCARLET_HEX = "ae2012"
    SCARLET_RGB = (174, 32, 18)
    SCARLET_RGB01 = tuple(c / 256 for c in SCARLET_RGB)

    RED_HEX = "9b2226"
    RED_RGB = (155, 34, 38)
    RED_RGB01 = tuple(c / 256 for c in RED_RGB)

    ALL_HEX = [
        BLACK_HEX,
        BLUE_HEX,
        CYAN_HEX,
        SEA_HEX,
        CREAM_HEX,
        GOLD_HEX,
        ORANGE_HEX,
        RUST_HEX,
        SCARLET_HEX,
        RED_HEX,
    ]

    ALL_RGB01 = [
        BLACK_RGB01,
        BLUE_RGB01,
        CYAN_RGB01,
        SEA_RGB01,
        CREAM_RGB01,
        GOLD_RGB01,
        ORANGE_RGB01,
        RUST_RGB01,
        SCARLET_RGB01,
        RED_RGB01,
    ]
    return (
        ALL_HEX,
        ALL_RGB01,
        BLACK_HEX,
        BLACK_RGB,
        BLACK_RGB01,
        BLUE_HEX,
        BLUE_RGB,
        BLUE_RGB01,
        CREAM_HEX,
        CREAM_RGB,
        CREAM_RGB01,
        CYAN_HEX,
        CYAN_RGB,
        CYAN_RGB01,
        GOLD_HEX,
        GOLD_RGB,
        GOLD_RGB01,
        ORANGE_HEX,
        ORANGE_RGB,
        ORANGE_RGB01,
        RED_HEX,
        RED_RGB,
        RED_RGB01,
        RUST_HEX,
        RUST_RGB,
        RUST_RGB01,
        SCARLET_HEX,
        SCARLET_RGB,
        SCARLET_RGB01,
        SEA_HEX,
        SEA_RGB,
        SEA_RGB01,
    )


@app.cell
def _(pl):
    def expand_json_column(df, column: str):
        original_keys = set(df.columns)
        df = (
            df.with_columns(pl.col(column).str.json_decode())
            .with_columns(pl.col(column).name.prefix_fields(f"{column}."))
            .unnest(column)
        )
        new_keys = set(df.columns) - original_keys
        return df

    return (expand_json_column,)


if __name__ == "__main__":
    app.run()
