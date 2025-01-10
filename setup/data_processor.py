from collections import defaultdict
from scipy import stats
from datetime import datetime
import numpy as np
import json, os, copy

"""
Gacha type 1 = Normal
Gacha type 2 = New player
Gacha type 11 = Event character
Gacha type 12 = Event light cone

Event Warp Light Cone
Light Cone 4★: 6,600% untuk item 4★ secara keseluruhan (termasuk Karakter 4★)

Event Warp Karakter
Karakter 4★: 2,550% untuk Karakter 4★ secara keseluruhan
Light Cone 4★: 2,550% untuk Light Cone 4★ secara keseluruhan

Event Warp Light Cone
Item 4★ (termasuk Karakter atau Light Cone 4★): 14,800% (termasuk sistem pity)

Event Warp Karakter
Item 4★ (termasuk Karakter atau Light Cone 4★): 13,000% (termasuk sistem pity)

"""


GACHA_TYPE = ["1", "2", "11", "12"]


class GachaDataProcessor:
    def __init__(self, uid, rank_type):
        self.uid = uid
        self.rank_type = rank_type
        self.hsr_datasets = self.__load_file(f"./datasets/hsr_dataset.json")
        self.history = self.__load_file(f"./datasets/hsr_gch_{self.uid}.json")
        self.history = self.__sort_data_by_id(self.history)
        self.processed_data = self.__get_process_data()
        self.cleaned_data = self.__get_clean_data()
        self.pity_data = self.__get_gct_pity(self.rank_type)
        self.statistic = self.__get_stat(self.pity_data)
        self.feature_statistic = self.__get_feature_stat(self.pity_data)
        self.filtered_feature_statistic = self.__get_filtered_feature_stat()
        self.time_statistic = self.__get_time_stat(self.pity_data)
        self.filtered_time_statistic = self.__get_filtered_time_stat()
        self.rate = self.__get_pity_rate(self.statistic)
        self.last_pity = self.__get_last_pity(self.pity_data)
        self.current_pity = self.__get_current_pity(self.last_pity, self.history)

    def __sort_data_by_id(self, data):
        for key in data:
            data[key] = sorted(data[key], key=lambda x: int(x["id"]))
        return data

    def __safe_divide(self, numerator, denominator):
        return (numerator / denominator) if denominator != 0 else 0

    def __load_file(self, path):
        if os.path.exists(path):
            with open(path, "r") as file:
                return json.load(file)
        else:
            with open(path, "w") as file:
                json.dump({}, file, indent=4)
            return {}

    def get_rate(self, gct: str, pity: int):
        def calc_rate(base_rate: float, max_rate: float, pity: int):
            if pity > 80:
                return base_rate + (
                    (pity % (80 if gct in ["11", "12"] else 90)) / 10 * max_rate
                )
            return base_rate

        if gct in ["1", "2", "11"]:
            if gct == "11":
                rate = calc_rate(0.006, 0.016, pity)
            else:
                rate = calc_rate(0.006, 0.016, pity)
        elif gct == "12":
            rate = calc_rate(0.008, 0.0187, pity)
        else:
            rate = 0
        return rate

    def __get_process_data(self):
        class TotalStats:
            current_pity: int = 1
            total_warp: int = 1
            total_r5: int = 0
            total_r4: int = 0
            total_win_r5: int = 0
            total_lose_r5: int = 0
            total_guarented_r5: int = 0
            total_win_r4: int = 0
            total_lose_r4: int = 0
            total_guarented_r4: int = 0

        def process_r5_item(
            item: dict, banner: dict, stats: TotalStats, r5_rate_on: bool
        ):
            item_id = item["item_id"]

            if item_id in str(banner["rateup"]) and not r5_rate_on:
                status = 2
                r5_rate_on = False
                stats.total_win_r5 += 1
            elif item_id not in str(banner["rateup"]) and not r5_rate_on:
                status = 0
                r5_rate_on = True
                stats.total_lose_r5 += 1
            else:
                status = 1
                r5_rate_on = False
                stats.total_guarented_r5 += 1

            stats.total_r5 += 1
            return status, r5_rate_on

        def process_r4_item(
            item: dict, banner: dict, stats: TotalStats, r4_rate_on: bool
        ):
            item_id = item["item_id"]

            if item_id in str(banner["rateup4"]) and not r4_rate_on:
                status = 2
                r4_rate_on = False
                stats.total_win_r4 += 1
            elif item_id not in str(banner["rateup4"]) and not r4_rate_on:
                status = 0
                r4_rate_on = True
                stats.total_lose_r4 += 1
            else:
                status = 1
                r4_rate_on = False
                stats.total_guarented_r4 += 1

            stats.total_r4 += 1
            return status, r4_rate_on

        def rate(total_item, total_rate):
            return round(self.__safe_divide(total_rate, total_item), 2)

        def update_item_stats(item: dict, stats: TotalStats) -> dict:
            item.update(
                {
                    "total_warp": stats.total_warp,
                    "total_r5": stats.total_r5,
                    "total_r4": stats.total_r4,
                    "get_rate_r5": rate(stats.total_warp, stats.total_r5),
                    "get_rate_r4": rate(stats.total_warp, stats.total_r4),
                    "win_rate_r5": rate(stats.total_r5, stats.total_win_r5),
                    "lose_rate_r5": rate(stats.total_r5, stats.total_lose_r5),
                    "guarented_rate_r5": rate(stats.total_r5, stats.total_guarented_r5),
                    "win_rate_r4": rate(stats.total_r4, stats.total_win_r4),
                    "lose_rate_r4": rate(stats.total_r4, stats.total_lose_r4),
                    "guarented_rate_r4": rate(stats.total_r4, stats.total_guarented_r4),
                }
            )
            return item

        processed_data = copy.deepcopy(self.history)
        for gacha_type, items in processed_data.items():
            pity_count_r5 = pity_count_r4 = 1
            r5_rate_on = r4_rate_on = False
            stats = TotalStats()

            for item in items:
                item.update(
                    {
                        "pity_r5": pity_count_r5,
                        "pity_r4": pity_count_r4,
                        "status_r5": 0,
                        "status_r4": 0,
                        "is_r5_rate_on": r5_rate_on,
                        "is_r4_rate_on": r4_rate_on,
                    }
                )

                banner = self.hsr_datasets["Banners"].get(item["gacha_id"])
                # Process R5 items
                if item["rank_type"] == "5":
                    status_r5, r5_rate_on = process_r5_item(
                        item, banner, stats, r5_rate_on
                    )
                    item["status_r5"] = status_r5
                    pity_count_r5 = 1
                else:
                    pity_count_r5 += 1

                # Process R4 items
                if item["rank_type"] == "4":
                    status_r4, r4_rate_on = process_r4_item(
                        item, banner, stats, r4_rate_on
                    )
                    item["status_r4"] = status_r4
                    pity_count_r4 = 1
                else:
                    pity_count_r4 += 1

                item.update(
                    {
                        "get_rate_r5": self.__safe_divide(pity_count_r5, 90),
                        "get_rate_r4": self.__safe_divide(pity_count_r4, 10),
                    }
                )
                # Update item statistics
                update_item_stats(item, stats)
                stats.total_warp += 1

        return processed_data

    def __get_clean_data(self):
        new_data = []
        for gct, items in self.processed_data.items():
            for item in items:
                data = {}
                dtime = datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")
                data["id"] = item["id"]
                data["gacha_type"] = item["gacha_type"]
                data["gacha_id"] = item["gacha_id"]
                data["item_id"] = item["item_id"]
                data["item_type"] = item["item_type"]
                data["rank_type"] = int(item["rank_type"])
                data["total_warp"] = int(item["total_warp"])
                data["total_r5"] = int(item["total_r5"])
                data["get_rate_r5"] = float(item["get_rate_r5"])
                data["win_rate_r5"] = float(item["win_rate_r5"])
                data["lose_rate_r5"] = float(item["lose_rate_r5"])
                data["guarented_rate_r5"] = float(item["guarented_rate_r5"])
                data["total_r4"] = int(item["total_r4"])
                data["get_rate_r4"] = float(item["get_rate_r4"])
                data["win_rate_r4"] = float(item["win_rate_r4"])
                data["lose_rate_r4"] = float(item["lose_rate_r4"])
                data["guarented_rate_r4"] = float(item["guarented_rate_r4"])
                data["pity_r4"] = int(item["pity_r4"])
                data["pity_r5"] = int(item["pity_r5"])
                data["status_r4"] = int(item["status_r4"])
                data["status_r5"] = int(item["status_r5"])
                data["status_r5"] = int(item["status_r5"])
                data["status_r5"] = int(item["status_r5"])
                data["is_r4_rate_on"] = 1 if item["is_r4_rate_on"] else 0
                data["is_r5_rate_on"] = 1 if item["is_r5_rate_on"] else 0
                data["year"] = dtime.year
                data["month"] = dtime.month
                data["date"] = dtime.day
                data["weekday"] = dtime.weekday()
                data["hour"] = dtime.hour
                data["minute"] = dtime.minute
                data["datetime"] = item["time"]
                new_data.append(data)
        return new_data

    def __get_gct_pity(self, rank_type="5"):
        rateup_rank = "rateup" if rank_type == "5" else "rateup4"
        pity = []
        for gct, items in self.history.items():
            pity_count = 1
            rate_on = False
            unique_char = []
            unique_lc = []
            item_r4 = 0
            item_r3 = 0
            for item in items:
                item_id = item["item_id"]
                item_type = item["item_type"]
                if item["rank_type"] == "4":
                    item_r4 += 1
                elif item["rank_type"] == "3":
                    item_r3 += 1

                if item_id not in unique_char and item_type == "Character":
                    unique_char.append(item_id)
                if item_id not in unique_lc and item_type == "Light Cone":
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
                            "item_r4": item_r4,
                            "item_r3": item_r3,
                            "unique_character": unique_char,
                            "unique_light_cone": unique_lc,
                            "status": status if not gct == "1" else "Guarented",
                            "time": item["time"],
                        }
                    )
                    unique_char = []
                    unique_lc = []
                    item_r3 = 0
                    item_r4 = 0
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
                "total_r4": 0,
                "total_r3": 0,
                "win": 0,
                "lose": 0,
                "guarented": 0,
            }
        )

        for gc in gc_pity:
            gacha_type = gc["gacha_type"]
            pity = int(gc["pity"])
            pity_data[gacha_type]["pity"].append(pity)
            pity_data[gacha_type]["total_r4"] += gc["item_r4"]
            pity_data[gacha_type]["total_r3"] += gc["item_r3"]
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
                    "total_item_r5": len(data["pity"]),
                    "total_item_r4": data["total_r4"],
                    "total_item_r3": data["total_r3"],
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

    def __get_filtered_feature_stat(self):
        return [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "pity_data",
                    "pity_win_data",
                }
            }
            for item in self.feature_statistic
        ]

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

    def __get_filtered_time_stat(self):
        return [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "year_data",
                    "month_data",
                    "date_data",
                    "weekday_data",
                    "hour_data",
                    "minute_data",
                    "second_data",
                }
            }
            for item in self.time_statistic
        ]

    def __get_pity_rate(self, stat: list):
        res = []
        for data in stat:
            win_rate = self.__safe_divide(data["total_win"], data["total_item_r5"])
            lose_rate = self.__safe_divide(data["total_lose"], data["total_item_r5"])
            collection_rate = self.__safe_divide(
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
                        self.__safe_divide(
                            data["total_guarented"], data["total_item_r5"]
                        )
                        * 100,
                        2,
                    ),
                    "unique_character_rate": round(
                        self.__safe_divide(
                            data["total_unique_character"], data["total_warp"]
                        )
                        * 100,
                        2,
                    ),
                    "unique_light_cone_rate": round(
                        self.__safe_divide(
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
