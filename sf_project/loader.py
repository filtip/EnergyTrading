from pathlib import Path
import pandas as pd
import numpy as np

class DataConfig:
    """
    Základní konfigurace pro zpracování surových dat.

    Parameters:
        data_dir (Path): Cesta k adresáři s měsíčními parquet soubory.
        tz (str): Časová zóna, do které budou data převedena
        freq (str): Frekvence minutového gridu
        product_mapping (dict): Mapování původních product_id na sjednocené názvy
    """
    def __init__(self, data_dir: Path, tz: str = "Europe/Prague", freq: str = "1min", product_mapping: dict = None):
        self.data_dir = data_dir
        self.tz = tz
        self.freq = freq
        self.product_mapping = (product_mapping if product_mapping is not None else {"EM_1": "EM_M_1"})


class LoadRawData:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def load(self, month: str) -> pd.DataFrame:
        """
        - Funkce načte původní data pro konkrétní měsíc
        - Nastaví timestamp jako index, převede index na správně časové pásmo, sjednotí mapping pro product_id

        Parameters:
            month: str - měsíc, pro který chceš data načíst
        """
        file_path = self.cfg.data_dir / f"data_{month}.parquet"
        df = pd.read_parquet(file_path)
        df = df.set_index("ts")
        df.index = df.index.tz_convert(self.cfg.tz)

        #Sjednocení mappingu u product_id
        if "product_id" in df.columns:
            df["product_id"] = df["product_id"].replace(self.cfg.product_mapping)
        return df

    def build_minute_grid_index(self, idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Vytvoří minutová časový grid pro každý den v indexu

        - Každý den vytvoří minutový grid od první až do posledního pozorování (nevytváři index přes noc)

        :param idx: Časový index pro který chceš vytvořit grid
        :return: pd.DatetimeIndex: spojený časový grid přes všechny obchodní dny
        """
        days = pd.to_datetime(pd.Series(idx.date).unique()).sort_values()
        grids = []

        for day in days:
            day_mask = (idx.date == day.date())
            day_idx = idx[day_mask]
            start = day_idx.min().floor(self.cfg.freq)
            end = day_idx.max().floor(self.cfg.freq)
            grid = pd.date_range(start, end, freq=self.cfg.freq, tz=self.cfg.tz)
            grids.append(grid)

        minute_grid_index = grids[0].append(grids[1:])
        return minute_grid_index


class PricePanelBuilder:
    SENTINEL = -999999
    def __init__(self, loader: LoadRawData):
        self.loader = loader

    @staticmethod
    def last_with_nan_priority(x: pd.Series) -> float:
        """
        Funkce, která nastaví logiku, které pozorování bude zvoleno pro minutový grid (8:00:00 - 8:00:59 --> ?)

        - V případě, že je definovaná cena -> vezme poslední z dané minuty
        - Pokud v danou minutu není definovaná cena, ale existuje nějaké NaN pozorování -> return SENTINEL
        - V případě, že neexistuje žádná pozorování v dané minutě -> NaN (bude forward fillnuté)
        SENTINEL zaručuje, že minuty, kdy víme, že cenu neznáme (=NaN), tak nebude forward fillnutá dle předchozích hodnot
        SENTINEL je poté nahrazen zpět na cena = NaN
        """
        valid = x[~x.isna()]
        if len(valid) > 0:
            return valid.iloc[-1]
        if len(x) > 0:
            return PricePanelBuilder.SENTINEL
        return np.nan

    def build_month_price_per_product(self, month : str) -> dict[str, pd.DataFrame]:
        """
        Minutový grid:

        - Pokud se v dané minutě vyskytne alespoň jedna skutečná hodnota ceny,
          použije se poslední platná (non-naN) cena z této minuty.
        - Pokud se v dané minutě vyskytují pouze hodnoty NaN,
          výsledná hodnota za tuto minutu zůstává NaN.
        - Pokud se v dané minutě nevyskytne žádné pozorování,
          hodnota je doplněna pomocí forward fill z posledního předchozího
          pozorování (NaN nebo skutečné ceny).

        Struktura panelu:

        - Panel obsahuje všechny obchodní dny, které se vyskytují v původních datech.
        - Nejsou aplikována žádná časová omezení (každý den je zahrnut celý,
          od prvního po poslední tick).
        - Každý obchodní den začíná hodnotou NaN (nedochází k přenosu ceny přes noc).
        """
        data = self.loader.load(month)
        index = self.loader.build_minute_grid_index(idx=data.index)
        freq = self.loader.cfg.freq
        product_ids = data["product_id"].unique()
        result_dict = {}

        for pid in product_ids:
            data_product = data[data["product_id"] == pid]

            #Kontrola, že v měsíci máme pro daný product_id stále stejnou hodnotu contract_id (může být NaN)
            contract_id = data_product["contract_id"].unique()
            if len(contract_id) == 0:
                contract_id = np.nan
            elif len(contract_id) == 1:
                contract_id = contract_id.item()
            else:
                raise ValueError(f"Multiple contract IDs in a {month} for {pid}")

            #Vytvoří minutový grid dle konkrétní transformace + forward fill ceny
            data_minute = data_product["price"].resample(freq).agg(self.last_with_nan_priority)
            data_filled = data_minute.groupby(data_minute.index.date).ffill()
            data_filled = data_filled.reindex(index)

            #Některé minuty s NaN cenou mohly být forward fillnuté -> opět vrať cena = NaN
            data_filled = data_filled.replace(self.SENTINEL, np.nan)

            df_long = pd.DataFrame(
                {
                    "product_id": pid,
                    "contract_id": contract_id,
                    "price": data_filled,
                },
                index=index,
            )

            df_final = df_long.sort_index()
            df_final.index.name = "index"
            result_dict[pid] = df_final

        return result_dict

    def combine_months(self, months: list[str]) -> dict[str, pd.DataFrame]:
        """
        Vytvoří ceny kontraktů pro více měsíců -> pouze vícekrát spustí metodu: build_month_price_per_product()
        :param months: list s měsíci: ["march", "april",...]
        :return: df s cenami konktraktů v dictionary
        """
        result = {}
        for month in months:
            month_dict = self.build_month_price_per_product(month)
            for pid, df_month in month_dict.items():
                if pid in result:
                    result[pid] = pd.concat([result[pid], df_month])
                else:
                    result[pid] = df_month
            print(f"{month} is DONE")

        for pid in result:
            result[pid] = result[pid].sort_index()

        return result


class BidAskPanelBuilder:
    def __init__(self, loader: LoadRawData):
        self.loader = loader

    @staticmethod
    def first_valid_bidask(df: pd.DataFrame) -> pd.Series:
        """
        Funkce, která nastaví logiku, které pozorování bude zvoleno pro minutový grid (8:00:00 - 8:00:59 --> ?)
        - Vrátí první pozorování v dané minutě, kde znám obě hodnoty bid & ask. (jinak nelze obchodovat)
        - Pokud žádná taková kotace neexistuje, vrátí NaN.
        """
        mask = df["best_bid"].notna() & df["best_ask"].notna()
        valid = df.loc[mask]
        if not valid.empty:
            return valid.iloc[0]

        return pd.Series({"best_bid": np.nan, "best_ask": np.nan})

    def build_month_bidask(self, month : str) -> pd.DataFrame:
        """
        Vytvoří minutová data bid & ask pro všechny kontrakty pro vybraný měsíc

        - Každý kontrakt je zpracován samostatně
        - Každý den zvlášť (žádný overnight fill)
        - Pro minuty, kde máme pozorování vezmu první takové, kde znám (bid, ask) - reálné hodnoty
        - Pokud v dané minutě existují pozorování, ale (bid,ask) nemá obě reálné hodnoty pro žádné z pozorování-> (NaN,NaN)
        - pro forward fill použiješ poslední hodnotu v dané minutě: (NaN,non-NaN) = (NaN,NaN) pro forward fill, (real,real) = (real,real)

        - Výstup: DataFrame (M_1_bid, M_1_ask, ...), 80 sloupců
        """
        data = self.loader.load(month)
        month_index = self.loader.build_minute_grid_index(idx=data.index)
        freq = self.loader.cfg.freq
        product_ids = data['product_id'].unique()
        panel_dict = {}

        for product in product_ids:
            data_product = data[data["product_id"] == product]
            filled_days = []

            for day, df_day in data_product.groupby(data_product.index.date):
                df_clean = df_day.dropna(subset=["best_bid", "best_ask"])

                #Přeskoč, pokud nejsou data
                if df_day.empty or df_clean.empty:
                    continue

                day_idx = month_index[month_index.date == day]

                #Udělej forward fill pro všechny minuty, doplníš poslední předchozí informaci (žádný look-ahead)
                minute_bidask = pd.merge_asof(
                    pd.DataFrame(index=day_idx),
                    df_day[["best_bid", "best_ask"]],
                    left_index=True,
                    right_index=True,
                    direction="backward"
                )

                #V případě, že neznáš v danou minutu obě hodnoty bid/ask -> NaN, NaN
                partial = minute_bidask["best_bid"].isna() ^ minute_bidask["best_ask"].isna()
                minute_bidask.loc[partial, ["best_bid", "best_ask"]] = np.nan

                #Minuty, kdy jsi znal bid&ask -> bereš první pozorování z dané minuty
                first_minute = (df_clean[["best_bid", "best_ask"]].resample(freq).apply(self.first_valid_bidask))
                first_minute.index = first_minute.index.tz_convert(minute_bidask.index.tz)
                first_minute = first_minute.reindex(minute_bidask.index)

                #Spoj logiku prnvího pozorování ve známé minutě a forward fill pomocí posledního pozorování
                mask = first_minute["best_bid"].notna() & first_minute["best_ask"].notna()
                minute_bidask.loc[mask, ["best_bid", "best_ask"]] = first_minute.loc[mask, ["best_bid", "best_ask"]]
                filled_days.append(minute_bidask)

            if filled_days:
                full_product_df = pd.concat(filled_days)
                full_product_df.columns = [f"{product}_bid", f"{product}_ask"]
                panel_dict[product] = full_product_df

        panel = pd.concat(panel_dict.values(), axis=1)
        panel = panel.sort_index()
        panel.index.name = "index"
        return panel


    def combine_months(self, months: list[str]) -> pd.DataFrame:
        """
        Vytvoří bid&ask kontraktů pro více měsíců -> pouze vícekrát spustí metodu: build_month_bidask()
        :param months: list s měsíci: ["march", "april",...]
        :return: bid/ask konktraktů v jednom DataFrame
        """
        results = []
        for month in months:
            bid_ask_month = self.build_month_bidask(month)
            results.append(bid_ask_month)
            print(f"{month} is DONE")
        df_all = pd.concat(results).sort_index()
        return df_all


class PanelSaver:
    """
    Pouze slouží k uložení dataframe do .parquet
    """
    @staticmethod
    def save_price_long(panels: dict[str, pd.DataFrame], path: Path) -> None:
        """
        Uloží DataFrame obsahující sloupce: product_id,contract_id, price
        """
        df_all = pd.concat(panels.values(), axis=0).sort_index()
        df_all.to_parquet(path)
        print(f"Saved {path}")

    @staticmethod
    def save_price_panel(panels : dict[str, pd.DataFrame], path: Path) -> None:
        """
        Uloží panelová data s cenou pro jednotlivé kontrakty (product_id, price)
        """
        df_all = pd.concat(panels.values(), axis=0).sort_index()
        panel = df_all.pivot(columns="product_id", values="price")
        panel.columns.name = None
        panel.to_parquet(path)
        print(f"Saved {path}")

    @staticmethod
    def save_bidask_panel(panels: pd.DataFrame, path: Path) -> None:
        """
        Uloží panelová data bid/ask pro jednotlivé kontrakty [M_1_bid, M_1_ask, M_2_bid, M_2_ask...]
        """
        panels.index.name = "index"
        panels.to_parquet(path)
        print(f"Saved {path}")

