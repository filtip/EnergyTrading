import pandas as pd
from datetime import time
from datetime import datetime
from pathlib import Path


class Specification:
    def __init__(self, target_col: str, open_until: time, close_until: time):
        self.target_col = target_col
        self.open_until = open_until
        self.close_until = close_until


class DataBuilder:
    def __init__(self, spec: Specification, prices: pd.DataFrame, bid_ask: pd.DataFrame, prediction: pd.DataFrame):
        self.spec = spec
        self.prices = prices
        self.bid_ask = bid_ask
        self.prediction = prediction

    def prepare_dataset(self) -> pd.DataFrame:
        """
        Vytvoří dataframe s price, bid, ask, predikce pro daný kontrakt
        Vzhledem ke způsobu, jak byly vytvořeny ceny a bid/ask, tak
        predikci v konkrétní minutě porovnáš s bid/ask v následjující minutě
        Price vytvořena jako poslední cena v minutě
        bid/ask jako prvni hodnota v minutě

        :return: DataFrame with 4 columns (bid,ask,prediction,price)
        """
        bid_ask_subset = self.bid_ask[[f"{self.spec.target_col}_bid", f"{self.spec.target_col}_ask"]]
        prices_subset = self.prices[[self.spec.target_col]]
        df = pd.concat([bid_ask_subset, self.prediction, prices_subset], axis=1)
        df.columns = ["bid", "ask", "prediction", "price"]

        # Posuň sloupce prediction o minutu dopředu, neboť chceš porovnávat predikci s bid/ask, která jsou o minutu později (vychází to z vytvoření bid/ask dat)
        df[["prediction"]] = df[["prediction"]].shift(1)
        df = df.between_time("8:00", "18:00")
        return df


class BaseStrategy:
    @staticmethod
    def close_position(df_day, close_time_limit, entry_price, position, position_size):
        """
        Automaticky po close_time_limit uzavřu pozici za price.
        :param df_day: DataFrame se 4 sloupci ziskaný z DataBuilder.prepare_dataset()
        :param close_time_limit: čas, kdy chceš zavřít pozici
        :param entry_price: vstupní cena do obchodu
        :param position: rozlišuje dlouhou/krátkou pozici
        :param position_size: velikost pozice v obchodu
        :return: close_price, t_close, pnl, position_size
        """
        day = df_day.index[0].date()
        time_target = pd.Timestamp(datetime.combine(day, close_time_limit), tz=df_day.index.tz)

        after_close = df_day[["bid", "ask", "price"]].loc[time_target:]

        mask_bid_ask = after_close[["bid", "ask"]].notna().any(axis=1)
        mask_price = after_close["price"].notna()
        mask = mask_bid_ask | mask_price

        valid_rows = after_close[mask]

        if valid_rows.empty:
            return None, None, None

        first_valid_row = valid_rows.iloc[0]
        bid2, ask2, price2 = first_valid_row["bid"], first_valid_row["ask"], first_valid_row["price"]
        t_close = first_valid_row.name

        if pd.notna(price2):
            close_price = price2
        else:
            close_price = (bid2 + ask2) / 2

        pnl = (close_price - entry_price) * position * position_size
        return close_price, t_close, pnl


class SingleEntry:
    def __init__(self, spec: Specification, threshold: float):
        """
        :param threshold: hranice o kterou musí predikce přeskočit bid/ask
        """
        self.spec = spec
        self.threshold = threshold

    def open_position(self, bid, ask, prediction):
        """
        Rozhodne jestli vstoupit do shortu/longu, pokud je predikce odlišná od intervalu (bid,ask)
        Nastav threshold = o kolik se predikce musí lišit, abych vstoupil do obchodu
        Otevře vždy pozici o velikosti 1.
        Otevře pozici nejpozději do open_until
        :return: position, entry_price, trade_type, position_size
        """

        if pd.notna(bid) and pd.notna(ask) and pd.notna(prediction):
            if prediction + self.threshold < bid:
                return -1, bid, "Short", 1
            if prediction - self.threshold > ask:
                return 1, ask, "Long", 1
        return 0, None, None, 0

    def signal_for_day(self, df_day: pd.DataFrame) -> pd.DataFrame:
        """
        Tato strategie vstoupí do pozice maximálně jednou denně s velikostí pozice 1
        Postupně v každé minutě kontroluje, jestli predikce leží v (bid,ask)
        Pokud predikce leží mimo interval o threshold -> vstoupí do pozice
        Poté se pozice automaticky zavře v 11h (později, pokud neznám price)

        :param df_day: DataFrame se 4 sloupci ziskaný z DataBuilder.prepare_dataset()
        :return: DataFrame s realizovaným obchodem pro daný kontrakt
        """
        day = df_day.index[0].date()
        open_until_target = pd.Timestamp(datetime.combine(day, self.spec.open_until), tz=df_day.index.tz)

        # Iteruješ po jednotlivých minutách do open_until
        for current_time, row in df_day.loc[:open_until_target].iterrows():
            bid, ask, price, prediction = row["bid"], row["ask"], row["price"], row["prediction"]

            #Zkus otevřít pozici a následně zavřít
            position, entry_price, trade_type, position_size = self.open_position(bid, ask, prediction)
            if position != 0:
                t_open = current_time
                close_price, t_close, pnl = BaseStrategy.close_position(df_day, self.spec.close_until, entry_price, position, position_size)

                trade = {
                    "threshold": self.threshold,
                    "contract": self.spec.target_col,
                    "day": day,
                    "trade_type": trade_type,
                    "t_open": t_open,
                    "t_close": t_close,
                    "entry_price": entry_price,
                    "close_price": close_price,
                    "position_size": position_size,
                    "pnl": pnl
                }
                df_trade = pd.DataFrame([trade])
                return df_trade

        return pd.DataFrame()

class MultiEntry:
    def __init__(self, spec: Specification, thresholds: list[float]):
        """
        Zde specifikuj posloupnost čísel, která definuje velikost pozice pro vstup do obchodu
        param thresholds: rostoucí posloupnost levelů pro další vstupy (např. [0.1, 0.2, 0.3, ...])
        Délka listu určuje velikost maximálního vstupu
        """
        self.spec = spec
        self.thresholds = thresholds

    def open_position(self, bid, ask, prediction, current_size) -> tuple:
        """
        Funkce zkusí otevřít pozici pokud predikce leží mimo interval (bid,ask)
        Zohlední, jestli už během dne do nějaké pozice bylo vstoupeno
        Velikost pozice určuje podle toho, jak moc se liší predikce od intervalu a podle hladin threshold

        V dané minutě zkusí vstoupit do pozice. Pokud má vstoupit opakovaně, tak se musí
        predikce lišít i vyšší hladinu threshold.
        Funkce tedy během dne postupně vstupuje do pozic na základě nastavených hladin threshold
        :return: position (oznaceni pro long/short), entry_price, trade_type, open_new, used_threshold
        """

        if not (pd.notna(bid) and pd.notna(ask) and pd.notna(prediction)):
            return 0, None, None, 0, None

        base_threshold = self.thresholds[current_size]

        if prediction + base_threshold < bid:
            open_new = 0
            #Zkontroluje, jestli se predikce neliší o vyšší edge -> vstupuji do větši pozice
            for j in range(current_size, len(self.thresholds)):
                if prediction + self.thresholds[j] < bid:
                    open_new += 1

            last_index = current_size + open_new - 1
            used_threshold = self.thresholds[last_index]
            return -1, bid, "Short", open_new, used_threshold

        elif prediction - base_threshold > ask:
            open_new = 0
            # Zkontroluje, jestli se predikce neliší o vyšší edge -> vstupuji do větši pozice
            for j in range(current_size, len(self.thresholds)):
                if prediction - self.thresholds[j] > ask:
                    open_new += 1

            last_index = current_size + open_new - 1
            used_threshold = self.thresholds[last_index]
            return 1, ask, "Long", open_new, used_threshold

        return 0, None, None, 0, None


    def signal_for_day(self, df_day: pd.DataFrame) -> pd.DataFrame:
        """
        Tato strategie umožňuje vstoupit do obchodu vícekrát během dně do maximální pozice velikosti = max_size (dlouhé/krátké)
        Obchoduje pouze jeden obchodní den a jeden konkrétní kontrakt. V danou minutu lze vstoupit do větších pozic než pouze 1.
        :param df_day: DataFrame se 4 sloupci ziskaný z DataBuilder.prepare_dataset()
        :return: DataFrame s realizovanými obchody pro daný kontrakt v konkrétní den
        """
        max_size = len(self.thresholds)
        current_size = 0
        day = df_day.index[0].date()
        open_until_target = pd.Timestamp(datetime.combine(day, self.spec.open_until), tz=df_day.index.tz)
        all_trades = []

        # Iteruji přes minuty a snažím se otevřít obchod, jsou-li splněny podmínky pro vstup
        for current_time, row in df_day.loc[:open_until_target].iterrows():

            # Pokud už mám nakoupenou maximální pozici -> už dále nevstupuji
            if current_size >= max_size:
                break

            # Zkusí otevřít pozici
            bid, ask, price, prediction = row["bid"], row["ask"], row["price"], row["prediction"]
            position, entry_price, trade_type, position_size, used_threshold = self.open_position(bid, ask, prediction, current_size)

            # Pokud se pozice neotevřela, pokračuji další iterací
            if position == 0:
                continue

            # Zavřu právě otevřenou pozici
            t_open = current_time
            current_size += position_size
            close_price, t_close, pnl = BaseStrategy.close_position(df_day, self.spec.close_until, entry_price, position, position_size)

            trade = {
                "threshold": used_threshold,
                "contract": self.spec.target_col,
                "day": day,
                "trade_type": trade_type,
                "t_open": t_open,
                "t_close": t_close,
                "entry_price": entry_price,
                "close_price": close_price,
                "new_opened": position_size,
                "total_opened_for_now": current_size,
                "pnl": pnl
            }

            all_trades.append(trade)

        return pd.DataFrame(all_trades)


class Backtester:
    def __init__(self, data_builder: DataBuilder, strategy):
        """
        :param data_builder:
        :param strategy: zviol strategii, kterou chceš použít (multi/single)
        """
        self.data_builder = data_builder
        self.strategy = strategy

    def run_contract(self) -> pd.DataFrame:
        """
        Spustí algoritmus pro všechny obchodní dny v datech se zvolenou strategii
        :return: Dataframe s realizovanými obchody během všech dní obsaženým v datech
        """
        df = self.data_builder.prepare_dataset()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_index()
        all_trades = []

        for _, df_day in df.groupby(df.index.date):
            trades_day = self.strategy.signal_for_day(df_day)
            if trades_day is not None and not trades_day.empty:
                all_trades.append(trades_day)

        return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()





def run_multiple_contracts(prices, bid_ask, contracts, dict_thresholds, open_until, close_until, prediction_path) -> pd.DataFrame:
    """
    Funkce která kombinuje třídy definované v tomto souboru.
    Funguje pro single/multi (podle toho jaký dostane dict_thresholds)

    :param prices: minutove ceny (in sample / out-of-sample
    :param bid_ask: bid_ask (in sample/ out of sample)
    :param contracts: kontrakty, které chceš zobchodovat
    :param dict_thresholds: (toto rozliší, jaká startegie se použije. Pokud dict[key:list] -> multi-entry)
    :param open_until: do kdy budeš otevírat pozice
    :param close_until: čas zavření obchodu
    :param prediction_path: cesta k predikcim
    :return: Realizované obchody pro všechny konktrakty přes celé obchodní období
    """
    all_trades = []
    prediction_path = Path(prediction_path)

    for contract in contracts:
        print(f"Processing {contract}...")

        pred_file = prediction_path / f"{contract}_predictions.parquet"
        if not pred_file.exists():
            print(f"  -> Missing prediction file, skipping.")
            continue

        prediction = pd.read_parquet(pred_file)

        spec = Specification(contract, open_until, close_until)
        builder = DataBuilder(spec, prices, bid_ask, prediction)

        val = dict_thresholds[contract]
        if isinstance(val, dict) and "optimal_threshold" in val:
            strategy = MultiEntry(spec, val["optimal_threshold"])
        else:
            strategy = SingleEntry(spec, float(val))


        backtest = Backtester(builder, strategy)
        trades = backtest.run_contract()
        trades["day"] = pd.to_datetime(trades["day"])


        if not trades.empty:
            all_trades.append(trades)

    return pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()


#SPUŠTĚNÍ JE V: strategy.py