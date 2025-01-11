from matplotlib.lines import Line2D
from .data_processor import GachaDataProcessor
from .equation_processor import EquationProcessor
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import copy


class VisualizeData:
    def __init__(self, uid, rank_type):
        self.data = GachaDataProcessor(uid, rank_type)
        self.equation = EquationProcessor(uid, rank_type)

    def distribute_item(self):
        gacha_types = [entry["gacha_type"] for entry in self.data.statistic]
        categories = [
            "total_unique_character",
            "total_unique_light_cone",
        ]
        values = {
            cat: [entry[cat] for entry in self.data.statistic] for cat in categories
        }
        x = np.arange(len(gacha_types))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (cat, vals) in enumerate(values.items()):
            bars = ax.bar(
                x + i * width,
                vals,
                width,
                label=(cat.capitalize().replace("_", " ")),
            )
            for bar in bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval / 2,
                    f"{int(yval)}",
                    ha="center",
                    va="center",
                    color="white",
                )

        ax.set_title("Distribution of Items")
        ax.set_xlabel("Gacha Type")
        ax.set_ylabel("Total")
        ax.set_xticks(x + width)
        ax.set_xticklabels(gacha_types)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        plt.tight_layout()
        plt.show()

    def pity_statistic(self):
        gacha_types = [entry["gacha_type"] for entry in self.data.feature_statistic]
        pity_data = [entry["pity_data"] for entry in self.data.feature_statistic]

        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(pity_data, patch_artist=True, showmeans=True)

        ax.set_title("Pity Statistics")
        ax.set_xlabel("Gacha Type")
        ax.set_ylabel("Pity Value")
        ax.set_xticks(range(1, len(gacha_types) + 1))
        ax.set_xticklabels(gacha_types)

        colors = ["skyblue", "lightgreen", "lightcoral", "green"]
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[0])
        for cap in bp["caps"]:
            cap.set(color=colors[2], linewidth=2)
        for median in bp["medians"]:
            median.set(color=colors[1], linewidth=3)
        for mean in bp["means"]:
            mean.set(color=colors[3], markersize=5)

        for i, median in enumerate(bp["medians"]):
            median_value = median.get_ydata()[0]
            ax.text(
                i + 1,
                median_value,
                f"{median_value:.2f}",
                ha="center",
                va="center",
            )

        for i, mean in enumerate(bp["means"]):
            mean_value = mean.get_ydata()[0]
            ax.text(
                i + 1,
                mean_value,
                f"{mean_value:.2f}",
                ha="center",
                va="center",
            )

        legend_labels = ["Min/Max", "Interquartile Range", "Median", "Mean"]
        legend_handles = [
            Line2D([0], [0], color=colors[2], lw=4, label=legend_labels[0]),
            Line2D([0], [0], color=colors[0], lw=4, label=legend_labels[1]),
            Line2D([0], [0], color=colors[1], lw=4, label=legend_labels[2]),
            Line2D([0], [0], color=colors[3], lw=4, label=legend_labels[3]),
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

    def pity_rate(self):
        gacha_types = [entry["gacha_type"] for entry in self.data.rate]
        win_rates = [entry["win_rate"] for entry in self.data.rate]
        lose_rates = [entry["lose_rate"] for entry in self.data.rate]
        guaranteed_rates = [entry["guarented_rate"] for entry in self.data.rate]

        total_rates = (
            np.array(win_rates) + np.array(lose_rates) + np.array(guaranteed_rates)
        )
        win_props = np.array(win_rates) / total_rates * 100
        lose_props = np.array(lose_rates) / total_rates * 100
        guaranteed_props = np.array(guaranteed_rates) / total_rates * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(gacha_types, win_props, label="Win", color="lightgreen")
        ax.barh(
            gacha_types, lose_props, left=win_props, label="Lose", color="lightcoral"
        )
        ax.barh(
            gacha_types,
            guaranteed_props,
            left=win_props + lose_props,
            label="Guaranteed",
            color="skyblue",
        )

        for i, (w, l, g) in enumerate(zip(win_props, lose_props, guaranteed_props)):
            # win
            ax.text(
                w / 2,
                i,
                f"{w:.1f}%",
                ha="center",
                va="center",
                alpha=0 if w == 0 else 1,
            )

            # lose
            ax.text(
                w + l / 2,
                i,
                f"{l:.1f}%",
                ha="center",
                va="center",
                alpha=0 if l == 0 else 1,
            )

            # guarenteed
            ax.text(
                w + l + g / 2,
                i,
                f"{g:.1f}%",
                ha="center",
                va="center",
                alpha=0 if g == 0 else 1,
            )

        ax.set_title("Win/Lose/Guaranteed Rates")
        ax.set_xlabel("Proportion in percentage")
        ax.set_ylabel("Gacha Type")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        plt.tight_layout()
        plt.show()

    def frequency_event_by_week(self):
        weekday_map = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }

        def draw_diagram(flat_weekdays, flat_hours, gacha_type):
            heatmap_data = pd.DataFrame({"weekday": flat_weekdays, "hour": flat_hours})
            heatmap_pivot = (
                heatmap_data.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
            )

            heatmap_pivot.index = heatmap_pivot.index.map(weekday_map)
            all_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            heatmap_pivot = heatmap_pivot.reindex(all_days, fill_value=0)
            all_hours = list(range(24))
            heatmap_pivot = heatmap_pivot.reindex(columns=all_hours, fill_value=0)

            plt.figure(figsize=(16, 6))
            sns.heatmap(
                heatmap_pivot, annot=True, cmap="YlGnBu", fmt="d", linewidths=0.5
            )
            plt.title(
                f"★5 Frequency for Gacha Type {gacha_type}"
                if gacha_type != "All"
                else "★5 Frequency for All Gacha Types"
            )
            plt.xlabel("Hour of Day")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()

        all_weekday_data = []
        all_hour_data = []

        for gacha in self.data.time_statistic:
            gacha_type = gacha["gacha_type"]
            weekday_data = gacha["weekday_data"]
            hour_data = gacha["hour_data"]

            flat_weekdays = [weekday for weekday in weekday_data]
            flat_hours = [hour for hour in hour_data]
            all_weekday_data.extend(flat_weekdays)
            all_hour_data.extend(flat_hours)
            draw_diagram(flat_weekdays, flat_hours, gacha_type)
        draw_diagram(all_weekday_data, all_hour_data, "All")

    def pity_change(self):
        df = pd.DataFrame(self.data.cleaned_data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by="datetime")
        grouped = df.groupby("gacha_type")["pity_r5"].sum().reset_index()

        plt.figure(figsize=(8, 6))
        for gacha_type in grouped["gacha_type"].unique():
            gacha_data = df[df["gacha_type"] == gacha_type]
            plt.plot(
                gacha_data["datetime"],
                gacha_data["pity_r5"],
                label=f"Gacha Type {gacha_type}",
            )
            for i in range(len(gacha_data)):
                if gacha_data["rank_type"].iloc[i] == 5:
                    plt.annotate(
                        gacha_data["pity_r5"].iloc[i],
                        (gacha_data["datetime"].iloc[i], gacha_data["pity_r5"].iloc[i]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                        fontsize=9,
                    )

        plt.title("Pity Changes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Pity")
        plt.grid(True)
        plt.legend(title="Gacha Type")
        plt.xticks(rotation=30)

        plt.tight_layout()
        plt.show()

    def check_correlation(self, except_col: list):
        df = pd.DataFrame(copy.deepcopy(self.data.cleaned_data))
        for col in except_col:
            df = df.drop(col, axis=1)
        correlation_matrix = df.corr()
        plt.figure(figsize=(18, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            vmin=-1,
            vmax=1,
        )
        plt.title(f"Correlation Matrix Heatmap")
        plt.show()
