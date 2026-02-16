import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_statistics(trades_all):
    """
    Input je dataframe s realizovanými obchody pro několik různých hladin threshold
    Dataframe musí obsahovat sloupce s threshold, contract
    Funkce napočítá základní statistiky: pnl, number_of_trades, number_of_shorts/longs
    :return:
    """
    thresholds = trades_all["threshold"].unique()
    thresholds = np.sort(thresholds)
    contracts = trades_all["contract"].unique()

    all_results = {}

    for contract in contracts:
        results = {}

        for threshold in thresholds:

            trades = trades_all[
                (trades_all["contract"] == contract) &
                (trades_all["threshold"] == threshold)
            ]

            number_of_trades = len(trades)

            if number_of_trades > 0:
                pnl = trades["pnl"].sum()
                avg_pnl = pnl / number_of_trades
                number_of_longs = (trades.trade_type == "Long").sum()
                number_of_shorts = (trades.trade_type == "Short").sum()
            else:
                pnl = 0
                avg_pnl = 0
                number_of_longs = 0
                number_of_shorts = 0

            results[threshold] = {
                "pnl": pnl,
                "number_of_trades": number_of_trades,
                "number_of_longs": number_of_longs,
                "number_of_shorts": number_of_shorts,
                "average_pnl": avg_pnl,
            }


        all_results[contract] = pd.DataFrame(results).T

    return all_results

def plot_statistics(statistics):
    """
    Funkce vytvoří 3 grafy pro každý kontrakt
    Jedná se o PnL v závislosti na threshold
    """
    for contract, results_df in statistics.items():
        results_df = results_df.sort_index()
        results_df.index.name = "threshold"

        #Threshold vs average PnL
        plt.figure(figsize=(8, 4))
        plt.plot(results_df.index, results_df["average_pnl"], marker="o")
        plt.title(f"Average PnL vs Threshold – {contract}")
        plt.xlabel("Threshold")
        plt.ylabel("Average PnL")
        plt.grid(True)
        plt.show()

        #Threshold vs total PnL
        plt.figure(figsize=(8, 4))
        plt.plot(results_df.index, results_df["pnl"], marker="o", color="orange")
        plt.title(f"Total PnL vs Threshold – {contract}")
        plt.xlabel("Threshold")
        plt.ylabel("Total PnL")
        plt.grid(True)
        plt.show()

        #Threshold vs number of trades
        plt.figure(figsize=(8, 4))
        plt.plot(results_df.index, results_df["number_of_trades"], marker="o", color="green")
        plt.title(f"Number of trades vs Threshold – {contract}")
        plt.xlabel("Threshold")
        plt.ylabel("Number of trades")
        plt.grid(True)
        plt.show()


def calculate_total_pnl(trades):
    #M -> 720$
    #QY -> 2160$
    #Y -> 8640$

    EM = ["EM_M_1"]
    M = ["GAS_M_1","GAS_M_2","M_1","M_2","M_3","M_4","M_5","IT_M_1"]
    Q = ["QY_1","QY_2","QY_3","QY_4","QY_5","QY_6","IT_QY_1"]
    Y = ["Y_1","Y_2"]

    results = {}
    total_sum = 0

    for target in EM:
        pnl = trades[trades["contract"] == target]["pnl"].sum()
        results[target] = pnl * 1000
        total_sum += results[target]

    for target in M:
        pnl = trades[trades["contract"] == target]["pnl"].sum()
        results[target] = pnl * 720
        total_sum += results[target]

    for target in Q:
        pnl = trades[trades["contract"] == target]["pnl"].sum()
        results[target] = pnl * 2160
        total_sum += results[target]

    for target in Y:
        pnl = trades[trades["contract"] == target]["pnl"].sum()
        results[target] = pnl * 8640
        total_sum += results[target]


    results["Total_PnL"] = total_sum

    df = pd.Series(results, name="PnL").to_frame()

    return df

def calculate_contract_statistics(trades):
    """
    Vypočítá základní statistiky pro každý kontrakt:
    - počet obchodů
    - počet long / short obchodů
    - celkový PnL
    - průměrný PnL na obchod
    """

    results = {}

    for contract in trades["contract"].unique():
        df = trades[trades["contract"] == contract]

        num_trades = len(df)
        num_longs = (df["trade_type"] == "Long").sum()
        num_shorts = (df["trade_type"] == "Short").sum()
        total_pnl = df["pnl"].sum()
        avg_pnl = total_pnl / num_trades if num_trades > 0 else 0

        results[contract] = {
            "num_trades": num_trades,
            "num_longs": num_longs,
            "num_shorts": num_shorts,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
        }

    return pd.DataFrame(results).T
