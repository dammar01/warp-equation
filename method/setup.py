from collections import defaultdict
from scipy import stats
from datetime import datetime
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json, os

GACHA_TYPE = ["1", "2", "11", "12"]


class GachaDataProcessor:
    def __init__(self, uid, rank_type):
        self.uid = uid
        self.rank_type = rank_type
        self.hsr_datasets = self.load_file(f"./datasets/hsr_dataset.json")
        self.history = self.load_file(f"./datasets/hsr_gch_{self.uid}.json")
        self.history = self.__sort_data_by_id(self.history)
        self.pity_data = self.__get_gct_pity(self.rank_type)
        self.statistic = self.__get_stat(self.pity_data)
        self.feature_statistic = self.__get_feature_stat(self.pity_data)
        self.time_statistic = self.__get_time_stat(self.pity_data)
        self.rate = self.__get_pity_rate(self.statistic)
        self.last_pity = self.__get_last_pity(self.pity_data)
        self.current_pity = self.__get_current_pity(self.last_pity, self.history)

    def __sort_data_by_id(self, data):
        for key in data:
            data[key] = sorted(data[key], key=lambda x: int(x["id"]))
        return data

    def safe_divide(self, numerator, denominator):
        return (numerator / denominator) if denominator != 0 else 0

    def load_file(self, path):
        if os.path.exists(path):
            with open(path, "r") as file:
                return json.load(file)
        else:
            with open(path, "w") as file:
                json.dump({}, file, indent=4)
            return {}

    def __get_gct_pity(self, rank_type="5"):
        rateup_rank = "rateup" if rank_type == "5" else "rateup4"
        pity = []
        for gct, items in self.history.items():
            pity_count = 1
            rate_on = False
            unique_char = []
            unique_lc = []
            for item in items:
                item_id = item["item_id"]
                item_type = item["item_type"]
                if item_id not in unique_char and item_type != "Character":
                    unique_char.append(item_id)
                elif item_id not in unique_lc and item_type != "Light Cone":
                    unique_lc.append(item_id)

                if item["rank_type"] == rank_type:
                    banner = self.hsr_datasets["Banners"].get(item["gacha_id"])
                    if item_id in str(banner[rateup_rank]) and not rate_on:
                        status = "Win"
                        rate_on = False
                    elif item_id not in str(banner[rateup_rank]) and not rate_on:
                        status = "Lose"
                        rate_on = True
                    else:
                        status = "Guarented"
                        rate_on = False
                    pity.append(
                        {
                            "gacha_type": gct,
                            "gacha_id": item["gacha_id"],
                            "item_id": item_id,
                            "pity": pity_count,
                            "unique_character": unique_char,
                            "unique_light_cone": unique_lc,
                            "status": status if not gct == "1" else "Guarented",
                            "time": item["time"],
                        }
                    )
                    pity_count = 1
                else:
                    pity_count += 1
        return sorted(pity, key=lambda x: x["time"])

    def __get_stat(self, gc_pity: list):
        pity_data = defaultdict(
            lambda: {
                "pity": [],
                "char_collection": [],
                "lc_collection": [],
                "total_warp": 0,
                "win": 0,
                "lose": 0,
                "guarented": 0,
            }
        )

        for gc in gc_pity:
            gacha_type = gc["gacha_type"]
            pity = int(gc["pity"])
            pity_data[gacha_type]["pity"].append(pity)
            pity_data[gacha_type]["char_collection"].extend(
                [
                    char
                    for char in gc["unique_character"]
                    if char not in pity_data[gacha_type]["char_collection"]
                ]
            )
            pity_data[gacha_type]["lc_collection"].extend(
                [
                    lc
                    for lc in gc["unique_light_cone"]
                    if lc not in pity_data[gacha_type]["lc_collection"]
                ]
            )
            pity_data[gacha_type]["total_warp"] += pity
            if gc.get("status") == "Win":
                pity_data[gacha_type]["win"] += 1
            elif gc.get("status") == "Lose":
                pity_data[gacha_type]["lose"] += 1
            else:
                pity_data[gacha_type]["guarented"] += 1

        res = []
        for gacha_type, data in pity_data.items():
            res.append(
                {
                    "gacha_type": gacha_type,
                    "total_item": len(data["pity"]),
                    "total_unique_character": len(data["char_collection"]),
                    "total_unique_light_cone": len(data["lc_collection"]),
                    "total_win": data["win"],
                    "total_lose": data["lose"],
                    "total_guarented": data["guarented"],
                    "total_warp": data["total_warp"],
                }
            )
        return sorted(res, key=lambda x: x["gacha_type"])

    def __get_feature_stat(self, gc_pity: list):
        pity_data = defaultdict(
            lambda: {
                "pity": [],
                "pity_win": [],
                "win_streak": 0,
                "lose_streak": 0,
            }
        )
        for gc in gc_pity:
            gacha_type = gc["gacha_type"]
            pity_data[gacha_type]["pity"].append(int(gc["pity"]))
            status = gc.get("status")
            if "current_ws" not in pity_data[gacha_type]:
                pity_data[gacha_type]["current_ws"] = 0
                pity_data[gacha_type]["current_ls"] = 0
            if status == "Win":
                pity_data[gacha_type]["current_ws"] += 1
                pity_data[gacha_type]["current_ls"] = 0
                pity_data[gacha_type]["win_streak"] = max(
                    pity_data[gacha_type]["win_streak"],
                    pity_data[gacha_type]["current_ws"],
                )
                pity_data[gacha_type]["pity_win"].append(int(gc["pity"]))
            elif status == "Lose":
                pity_data[gacha_type]["current_ls"] += 1
                pity_data[gacha_type]["current_ws"] = 0
                pity_data[gacha_type]["lose_streak"] = max(
                    pity_data[gacha_type]["lose_streak"],
                    pity_data[gacha_type]["current_ls"],
                )

        res = []
        for gacha_type, data in pity_data.items():
            total_pity = data["pity"]
            if total_pity:
                avg_pity = round(np.mean(total_pity), 2)
                median_pity = round(np.median(total_pity), 2)
                mode_pity = int(stats.mode(total_pity, keepdims=True).mode[0])
            else:
                avg_pity = median_pity = mode_pity = 0

            res.append(
                {
                    "gacha_type": gacha_type,
                    "pity_data": data["pity"],
                    "pity_win_data": data["pity_win"],
                    "win_streak": data["win_streak"],
                    "lose_streak": data["lose_streak"],
                    "min_pity": np.min(data["pity"]),
                    "max_pity": np.max(data["pity"]),
                    "min_pity_win": (
                        np.min(data["pity_win"])
                        if len(data["pity_win"]) > 0
                        else np.min(data["pity"])
                    ),
                    "max_pity_win": (
                        np.max(data["pity_win"])
                        if len(data["pity_win"]) > 0
                        else np.max(data["pity"])
                    ),
                    "avg_pity": avg_pity,
                    "median_pity": median_pity,
                    "mode_pity": mode_pity,
                }
            )
        return sorted(res, key=lambda x: x["gacha_type"])

    def __get_time_stat(self, gc_pity: list):
        pity_data = defaultdict(
            lambda: {
                "avg_year": [],
                "avg_month": [],
                "avg_date": [],
                "avg_weekday": [],
                "avg_hour": [],
                "avg_minute": [],
                "avg_second": [],
            }
        )

        for gc in gc_pity:
            gacha_type = gc["gacha_type"]
            dtime = datetime.strptime(gc["time"], "%Y-%m-%d %H:%M:%S")
            pity_data[gacha_type]["avg_year"].append(dtime.year)
            pity_data[gacha_type]["avg_month"].append(dtime.month)
            pity_data[gacha_type]["avg_date"].append(dtime.day)
            pity_data[gacha_type]["avg_weekday"].append(dtime.weekday())
            pity_data[gacha_type]["avg_hour"].append(dtime.hour)
            pity_data[gacha_type]["avg_minute"].append(dtime.minute)
            pity_data[gacha_type]["avg_second"].append(dtime.second)
        res = []
        for gacha_type, data in pity_data.items():
            res.append(
                {
                    "gacha_type": gacha_type,
                    "year_data": data["avg_year"],
                    "month_data": data["avg_month"],
                    "date_data": data["avg_date"],
                    "weekday_data": data["avg_weekday"],
                    "hour_data": data["avg_hour"],
                    "minute_data": data["avg_minute"],
                    "second_data": data["avg_second"],
                    "avg_year": (
                        round(np.mean(data["avg_year"]), 2) if data["avg_year"] else 0
                    ),
                    "avg_month": (
                        round(np.mean(data["avg_month"]), 2) if data["avg_month"] else 0
                    ),
                    "avg_date": (
                        round(np.mean(data["avg_date"]), 2) if data["avg_date"] else 0
                    ),
                    "avg_weekday": (
                        round(np.median(data["avg_weekday"]), 2)
                        if data["avg_weekday"]
                        else 0
                    ),
                    "avg_hour": (
                        round(np.mean(data["avg_hour"]), 2) if data["avg_hour"] else 0
                    ),
                    "avg_minute": (
                        round(np.mean(data["avg_minute"]), 2)
                        if data["avg_minute"]
                        else 0
                    ),
                    "avg_second": (
                        round(np.mean(data["avg_second"]), 2)
                        if data["avg_second"]
                        else 0
                    ),
                }
            )
        return sorted(res, key=lambda x: x["gacha_type"])

    def __get_pity_rate(self, stat: list):
        res = []
        for data in stat:
            win_rate = self.safe_divide(data["total_win"], data["total_item"])
            lose_rate = self.safe_divide(data["total_lose"], data["total_item"])
            collection_rate = self.safe_divide(
                data["total_unique_light_cone"] + data["total_unique_character"],
                len(self.hsr_datasets["Character"])
                + len(self.hsr_datasets["Light Cone"]),
            )
            res.append(
                {
                    "gacha_type": data["gacha_type"],
                    "win_rate": round(win_rate * 100, 2),
                    "lose_rate": round(lose_rate * 100, 2),
                    "guarented_rate": round(
                        self.safe_divide(data["total_guarented"], data["total_item"])
                        * 100,
                        2,
                    ),
                    "unique_character_rate": round(
                        self.safe_divide(
                            data["total_unique_character"], data["total_warp"]
                        )
                        * 100,
                        2,
                    ),
                    "unique_light_cone_rate": round(
                        self.safe_divide(
                            data["total_unique_light_cone"], data["total_warp"]
                        )
                        * 100,
                        2,
                    ),
                    "collection_rate": round(collection_rate * 100, 2),
                }
            )
        return res

    def __get_last_pity(self, gc_pity: list):
        last_pity = {}
        for gc in gc_pity:
            gacha_type = gc["gacha_type"]
            if (
                gacha_type not in last_pity
                or gc["time"] > last_pity[gacha_type]["time"]
            ):
                last_pity[gacha_type] = {
                    "gacha_type": gc["gacha_type"],
                    "gacha_id": gc["gacha_id"],
                    "item_id": gc["item_id"],
                    "last_pity": gc["pity"],
                    "status": gc["status"],
                    "time": gc["time"],
                }
        return sorted(list(last_pity.values()), key=lambda x: x["gacha_type"])

    def __get_current_pity(self, gc_last_pity: list, gct_data: dict):
        last_pity_map = {pity["gacha_type"]: pity for pity in gc_last_pity}
        pity_results = []
        for gacha_type, items in gct_data.items():
            if len(items) == 0:
                continue
            pity_count = 0
            last_time = last_pity_map.get(gacha_type, {}).get("time", "")
            last_status = last_pity_map.get(gacha_type, {}).get("status", "")
            cr_time = last_time
            for item in items:
                if item["time"] > last_time:
                    pity_count += 1
                    cr_time = item["time"]
            pity_results.append(
                {
                    "gacha_type": gacha_type,
                    "current_pity": pity_count,
                    "status": "Rate On" if last_status == "Lose" else "Rate Off",
                    "time": cr_time,
                }
            )
        return sorted(pity_results, key=lambda x: x["gacha_type"])


class VisualizeData:
    def __init__(self, data: GachaDataProcessor):
        self.data = data

    def distribute_item(self):
        gacha_types = [entry["gacha_type"] for entry in self.data.statistic]
        categories = ["total_item", "total_unique_character", "total_unique_light_cone"]
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
                label=(
                    cat.capitalize().replace("_", " ")
                    if cat != "total_item"
                    else cat.capitalize().replace("_", " ") + " ★5"
                ),
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

        ax.set_title("Distribution of Items by Gacha Type")
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

        ax.set_title("Pity Statistics by Gacha Type")
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

        ax.set_title("Win/Lose/Guaranteed Rates by Gacha Type")
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
