import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


class Specification:
    """
    Nastavení targetu a jeho prediktorů, které jsou využity k tréninku + predikci
    """
    def __init__(self, target_col: str, predictor_cols: list[str]):
        self.target_col = target_col
        self.predictor_cols = predictor_cols
        self.cols = self.predictor_cols + [self.target_col]


class DataTransformation:
    """
    V této třídě jsou metody, které počítají overnight gaps
    """
    def __init__(self, spec: Specification, interval_length: int = 20, number_of_observations: int = 15):
        """
        :param spec: instance ze Specification
        :param interval_lenght: Délka časového okna ze kterého chceš počítat průměr ceny
        :param number_of_observations: Minimálmí počet pozorování nutný pro výpočet průměru v daném okně
        """
        self.spec = spec
        self.interval_length = interval_length
        self.number_of_observations = number_of_observations

    def morning_average(self, data, after_time = "10:00", up_to_time = "18:00") -> pd.DataFrame:
        """
        :param data: minutová data s cenou kontraktů
        :param after_time: Nastav počátek časového okna k hledání ranních pozorování
        :param up_to_time: Nastav nejpozdější termín časového okna

        Funkce každý den spočítá ranní průměrnou cenu. Funkce začíná iterovat časové okno hned od after_time (10:00-10:20)
        Průměr je brán z prvního časového okna délky=interval_lenth, které obsahuje alespoň počet pozorování = number_of_observations
        a začína později něž after_time.
        :return: DataFrame s dopolední průměrnou cenou kontraktů predictor_cols, target_col (každý obchodní den)
        """
        data = data[self.spec.cols].dropna(how='any').between_time(after_time, up_to_time)
        morning_prices = []
        days = pd.Series(data.index.date).unique()

        for day in days:
            data_day = data.loc[data.index.date == day]
            minutes = data_day.index

            # Po nastaveném času hledáš první interval, kde je dost pozorování
            for minute in minutes:
                data_window = data_day.loc[minute:minute + pd.Timedelta(minutes=self.interval_length)]
                if data_window.shape[0] >= self.number_of_observations:
                    morning_price = data_window.mean()
                    first_timestamp = data_window.index[0]

                    morning_price.name = first_timestamp
                    morning_prices.append(morning_price)
                    break

        prices = pd.DataFrame(morning_prices)
        return prices

    def afternoon_average(self, data, after_time = "8:00", up_to_time = "17:00") -> pd.DataFrame:
        """
        :param data: Minutová data
        :param after_time: Nastav poslední možnou iteraci
        :param up_to_time: Nastav čas od kdy iteruješ zpět

        Funkce spočítá každý den odpolední průměrné ceny. Hledá první časové okno před up_to_time (začne s 16:40-17:00),
        délky = interval_length a obsahuje alespoň počet pozorování = number_of_observations. Poté odpolední cena
        je průměrná cena z tohoto odpovídajícího časového okna.
        :return: DataFrame s odpolední průměrnou cenou kontraktů predictor_cols, target_col (každý obchodní den)
        """
        data = data[self.spec.cols].dropna(how='any').between_time(after_time, up_to_time)
        afternoon_prices = []
        days = pd.Series(data.index.date).unique()

        for day in days:
            data_day = data.loc[data.index.date == day]
            minutes = data_day.index

            # Po nastaveném času hledáš první interval, kde je dost pozorování
            for minute in minutes.sort_values(ascending=False):
                data_window = data_day.loc[(minute - pd.Timedelta(minutes=self.interval_length)):minute]
                if data_window.shape[0] >= self.number_of_observations:
                    afternoon_price = data_window.mean()
                    first_timestamp = data_window.index[0]

                    afternoon_price.name = first_timestamp
                    afternoon_prices.append(afternoon_price)
                    break

        prices = pd.DataFrame(afternoon_prices)
        return prices

    def calculate_overnight_gaps(self, data) -> pd.DataFrame:
        """
        :param data: Minutová data

        Tahle funkce spočítá deltu ranního pozorování a pozorování z předchozího odpoledne každého z predictor_cols a target, pokud existují
        Beru vždy rozdíl (ráno - předchozí odpoledne), (ráno - gap o dvě pozorování)
        Nebere delty napříč měsíci kvůli jinému mappingu
        :return: DataFrame s deltami přes noc (delta o 1 pozorování, delta o 2 pozorování)
        """
        morning_prices = self.morning_average(data)
        afternoon_prices = self.afternoon_average(data)
        months = pd.Series(data.index.month).unique()
        all_deltas = []

        for month in months:
            morning_month = morning_prices[morning_prices.index.month == month]
            afternoon_month = afternoon_prices[afternoon_prices.index.month == month]
            afternoon_days = afternoon_month.index.normalize()

            # Iteruj přes ranní pozorování a hledej k nim vždy poslední předchozí odpolední
            for index, row in morning_month.iterrows():
                morning_day = index.normalize()

                mask = afternoon_days < morning_day
                afternoon_previous = afternoon_month.loc[mask].sort_index()

                # Kontrola, kolik existuje předchozích odpoledních pozorování
                if len(afternoon_previous) == 0:
                    continue

                # Pokud existuje 1 přechozí pozorování -> máme jednu deltu
                if len(afternoon_previous) == 1:
                    previous1 = afternoon_previous.iloc[0]
                    delta = row - previous1
                    delta.name = index
                    all_deltas.append(delta)

                # Zde spočítáme vždy 2 delty
                else:
                    previous1 = afternoon_previous.iloc[-1]
                    previous2 = afternoon_previous.iloc[-2]
                    delta1 = row - previous1
                    delta2 = row - previous2

                    delta1.name = index
                    delta2.name = index

                    all_deltas.append(delta1)
                    all_deltas.append(delta2)

        all_diffs = pd.DataFrame(all_deltas)
        return all_diffs


class TrainModel:
    def __init__(self, spec: Specification, data_transformation: DataTransformation):
        self.spec = spec
        self.data_transformation = data_transformation
        self.model = None
        self.mse_train = None
        self.mse_test = None
        self.rsquared = None
        self.training_months = None

    def _fit(self, data: pd.DataFrame, prediction_month: str) -> tuple:
        """
        Pro konkrétí data, ktera tato metoda dostane natrénuje lineární model s targetem target_col a prediktory: predictor_cols
        Model je natrénován na všech měsících, které jsou v data bez prediction_month (one-month-out)

        :param data: minutová data (musí obsahovat sloupce target_col, predictor_cols)
        :param prediction_month: Měsíc, který chceš vytvářet predikce (nebude použit k tréninku)
        :return: natrénovaný model + charakteristika modelu
        """
        # Vyber pouze ceny kontraktů prediktorů a targetu
        X = data[self.spec.predictor_cols]
        y = data[self.spec.target_col]

        # Získej měsíční indexy
        periods = X.index.to_period("M")
        pred_period = pd.Period(prediction_month)

        train_mask = periods != pred_period  # Trénujeme na všech měsících kromě toho, který predikujeme
        test_mask = periods == pred_period  # Měsíc, který predikujeme

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Vybereme pouze pozorování, které mají definovano cenou (nejsou NaN)
        mask_train = X_train.notna().all(axis=1) & y_train.notna()
        mask_test = X_test.notna().all(axis=1) & y_test.notna()

        # Filtrujeme pouze pozorování, kde známe cenu všech kontraktů
        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]

        # SAFETY CHECK
        if X_train.shape[0] == 0:
            raise ValueError(f"ERROR: No training data for predict_month={prediction_month}")

        # Přidej intercept a natrénuj lineární model
        X_train = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train).fit()

        # Spočítej MSE na testovací sadě
        if X_test.shape[0] == 0:
            X_test = None
            y_test = None
            mse_test = None
        else:
            X_test = sm.add_constant(X_test)
            y_pred_test = model.predict(X_test)
            mse_test = mean_squared_error(y_test, y_pred_test)

        # Spočítej další metriky
        y_pred_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        rsquared = model.rsquared

        training_months = y_train.index.to_period("M").unique()
        return model, rsquared, mse_train, mse_test, training_months

    def fit_on_scenario(self, data: pd.DataFrame, prediction_month: str):
        """
        Model se natrénuje na všech datech bez dat z měsíce prediction_month
        Tato metoda vytvoři over_night gaps a pro daný scénář (target, predictors) natrénuje model
        :param data: minutová data s cenami kontraktů
        :param prediction_month: zvol měsíc, který chceš predikovat (model na něm nebude natrénován)
        """
        scenario = self.data_transformation.calculate_overnight_gaps(data)
        model, rsquared, mse_train, mse_test, training_months = self._fit(scenario, prediction_month)

        self.model = model
        self.mse_train = mse_train
        self.mse_test = mse_test
        self.rsquared = rsquared
        self.training_months = training_months


    def fit_on_scenario_out_of_sample(self, data_in: pd.DataFrame, data_out: pd.DataFrame, prediction_month: str):
        """
        Model se natrénuje na všech datech, které jsou před prediction_month (žádný look-ahead)
        :param data_in: minutová data - in sample
        :param data_out: minutová data - out of sample
        :param prediction_month: zvol měsíc, který chceš predikovat
        """
        data_all = pd.concat([data_in, data_out], axis = 0).sort_index()
        tz = data_all.index.tz
        start = pd.Period(prediction_month, freq="M").to_timestamp(how="start")
        start = start.tz_localize(tz)
        data_hist = data_all[data_all.index < start]

        scenario = self.data_transformation.calculate_overnight_gaps(data_hist)
        model, rsquared, mse_train, mse_test, training_months = self._fit(scenario, prediction_month)

        self.model = model
        self.mse_train = mse_train
        self.mse_test = mse_test
        self.rsquared = rsquared
        self.training_months = training_months


class InputsBuilder:
    """
    Tato připravuje data k finálním predikcím. K dopoledním cenám kontraktů hledám ceny z předchozího závěru dne, aby mohl spočítat delty
    Cena z předchozího dne se vyhlazuje, aby byla přesnější. Rozlišuje vytvoření dat pro první den v měsíci a zbytku měsíce (jiný mapping)
    """
    def __init__(self, spec: Specification):
        self.spec = spec

    def candidates_regular(self, data: pd.DataFrame, prediction_month: str) -> pd.DataFrame:
        """
        Vezme dopolední pozorování, kde vždy známe ceny prediktorů
        Budeme dělat predikce pouze do 12:00
        Vyřadíme z výsledného dataframe pozorování ze začátku měsíce, neboť taková predikce vychází
        z jiného mapingu

        :param data: untransformed data (minute data)
        :param predictor_cols: list of column names of predictors
        :param target_col: column name of target variable
        :param prediction_month: měsíc, který chceš predikovat

        :return: DataFrame s dopoledními minutami, kdy je definovaná cena prediktorů
        """
        data = data[self.spec.cols]
        data_predict_month = data.loc[prediction_month]

        if data_predict_month.empty:
            raise ValueError(f"No data for prediction in [{prediction_month}]")

        prediction_candidates = data_predict_month.between_time("7:00", "12:00")
        prediction_candidates = prediction_candidates.dropna(subset=self.spec.predictor_cols)

        # Toto pouze zajistí, aby nebyla predikce pro první dny v měsíci, musí být vytvořena jinak (jiný mapping)
        first_valid_observation = data_predict_month.dropna(subset=self.spec.cols)
        if not first_valid_observation.empty:
            first_valid_timestamp = first_valid_observation.index[0]
            cutoff_day = first_valid_timestamp.normalize() + pd.Timedelta(days=1)
            prediction_candidates = prediction_candidates.loc[prediction_candidates.index >= cutoff_day]

        return prediction_candidates

    def previous_regular(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Funkce spočte průměrnou cenu prediktorů a targetu z konce obchodního dne
        :return: DataFrame s průměrnými cenami na konci jednotlivých obchodních dní
        """
        data = data[self.spec.cols].dropna(how="any")
        days = pd.Series(data.index.date).unique()
        means = []

        # Každý den chceš spočítat průměrné ceny kontraktů ke konci dne
        for day in days:
            daily_data = data[data.index.date == day]

            if len(daily_data) == 0:
                continue
            if len(daily_data) >= 15:
                last_ts = daily_data.index[-1]
                ts_15 = daily_data.index[-15]

                if last_ts - ts_15 <= pd.Timedelta(hours=1):
                    window = daily_data.iloc[-15:]
                else:
                    window = daily_data.iloc[-5:]

            # Zde bere průměr celýho dne, rozhodně by bylo vhodné se na to podívat
            else:
                window = daily_data
            daily_last_mean = window.mean()
            daily_last_mean.name = daily_data.index[-1]
            means.append(daily_last_mean)

        last_previous_observations = pd.DataFrame(means).sort_index()
        return last_previous_observations


    def candidates_first_day(self, data_w_contract_id: pd.DataFrame, prediction_month: str) -> pd.DataFrame:
        """
        Vytvoří dataframe s ranními cenami první obchodní den společně s informaci o contract_id pro target, predictors

        :param data_w_contract_id -> minutová data společně s product_id, cotract_id, price
        :param prediction_month: měsíc, který chceš predikovat
        return: DataFrame s ranní cenou prediktorů a přislušnými contract_id pro target, predictors
        """
        # Vyber pouze prediction_month
        data_month = data_w_contract_id.loc[prediction_month]
        subset = data_month[data_month["product_id"].isin(self.spec.cols)]

        minute_index = pd.date_range(
            start=data_month.index.min().floor("min"),
            end=data_month.index.max().floor("min"),
            freq="1min",
            tz=data_month.index.tz
        )

        # Vytvoří df se cenou pro každé product_id z cols
        wide_prices = subset.pivot_table(
            index=subset.index,
            columns="product_id",
            values="price"
        )

        # Pro každý product_id vytvoř sloupec s contract_id.
        contract_wide = subset.pivot_table(
            index=subset.index,
            columns="product_id",
            values="contract_id",
            aggfunc="first"
        )

        # Sjednocení time index
        wide_prices = wide_prices.reindex(minute_index)
        contract_wide = contract_wide.reindex(minute_index)

        # Zde máš df se názvy sloupců = product_id a values = price
        df = wide_prices[self.spec.cols].copy()

        # Přidá sloupce s označením o jaké contract_id se jedná pro každý product_id
        for product in self.spec.cols:
            df[f"{product}_contract_id"] = contract_wide[f"{product}"]

        #Najdi první pozorování, kde znáš ceny všech konktraktu: prediktorů a targetu
        valid_target = df[self.spec.cols].dropna()
        first_ts = valid_target.index[0]
        day_first = first_ts.normalize()
        mask_days = df.index.normalize() <= day_first

        # Vezme všechny přechozí obchodní dny včetně prvního známého dne ceny všech cols
        prediction_candidates = df[mask_days]
        prediction_candidates = prediction_candidates.dropna(subset=self.spec.predictor_cols).between_time("7:00", "12:00")

        # Pokud neznáš hodnoty prediktorů dané dopoledne -> nelze predikovat (zvol jiné prediktory)
        if prediction_candidates.empty:
            print(f"empty prediction_candidates in {prediction_month} on first days")
            print("Unknown values of predictors between 7-12h")
            return pd.DataFrame()

        return prediction_candidates

    def previous_first_day(self, data_w_contract_id: pd.DataFrame, prediction_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Tahle funkce najde dle contract_id pozorování z konce přechozího měsíce, aby bylo možné napočítat delty s prvním dnem v měsíci
        Najde poslední pozorování z měsíce, kde existuje target&predictors.
        Spočítá průměr z posledních 15 hodnot, pokud tyto hodnoty leží v rozmezí 1h a jsou na konci dne (+ trochu složitější)

        :param data_w_contract_id: dataframe with contract_id, product_id, prices
        :param prediction_candidates: df s cenami prediktorů v časech, kdy chceš dělat predikci v první den (z metody candidates_first_day())
        :return: df se stabilnějším odhadem ceny prediktorů a targetu z posledního dne v měsíci
        """

        # Vytvoř mapping: product_id -> contract_id
        contract_id_mapping = {}
        for col in self.spec.cols:
            contract_id_mapping[col] = prediction_candidates[f"{col}_contract_id"].iloc[0]
        contract_ids = list(contract_id_mapping.values())

        data_filtered = data_w_contract_id[data_w_contract_id["contract_id"].isin(contract_ids)]

        # Vytvoř df s cenou a názvy sloupců contract_id
        wide = data_filtered.pivot_table(
            index=data_filtered.index,
            columns="contract_id",
            values="price"
        )

        # Přejmenuj sloupce v df wide dle mappingu -> nyní už mám názvy sloupců opět product_id
        rename_mapping = {v: k for k, v in contract_id_mapping.items()}
        wide = wide.rename(columns=rename_mapping)
        wide = wide[self.spec.cols]

        # Budeš počítat deltu vzhledem k pozorováním předchozím k prediction_candidates
        first_ts = prediction_candidates.index[0]
        day_first = first_ts.normalize()
        mask_days = wide.index.normalize() < day_first

        # Vyber jen ceny z předchozího měsíce
        previous_observations = wide.loc[mask_days]
        previous_observations = previous_observations.dropna(how="any").sort_index()

        # Kontrola, jestli existoval contract_id i minulý měsíc (občas nemáme k dispozici, př. M_5, EM_M_1,...)
        if previous_observations.empty:
            print(f"No prediction for first day {self.spec.target_col}, {first_ts}")
            return pd.DataFrame()

        # Počítáš přesnější cenu pomocí průměru několika hodnot z konce obchodního dne (cols)
        if len(previous_observations) >= 15:
            last_ts = previous_observations.index[-1]
            ts_15 = previous_observations.index[-15]

            if last_ts - ts_15 <= pd.Timedelta(hours=1):
                window = previous_observations.iloc[-15:]
            else:
                window = previous_observations.iloc[-5:]
        else:
            window = previous_observations

        last_mean = window.mean()
        last_mean.name = previous_observations.index[-1]

        # Vzhledem k tomuto pozorování budem počítat deltu (dopolední ceny - last_observation)
        last_observation = pd.DataFrame([last_mean], index=[last_mean.name])
        return last_observation



class Prediction:
    """
    Tato třída spočítá delty přes noc, vytvoří predikce těchto delt pro target pomocí již natrénovaného modelu
    """
    def __init__(self, trainer: TrainModel, inputs: InputsBuilder):
        self.trainer = trainer
        self.inputs = inputs
        self.spec = inputs.spec

    def predict_from_inputs(self, prediction_candidates: pd.DataFrame, previous_observations: pd.DataFrame) -> pd.Series:
        """
        Spočítá delty dopoledních cen vs odpolední cena předchozí obchodní den
        Pomocí modelu ze třídy TrainModel vytvoří predikce delt pro target

        :param prediction_candidates: výstup z candidates_regular() nebo candidates_first_day()
        :param previous_observations: výstup z previous_regular() nebo previous_first_day()
        :return: pd.Series s predikcemi pre všechny spočtené delty
        """
        if prediction_candidates.empty or previous_observations.empty:
            return pd.Series(dtype=float)

        model = self.trainer.model

        miss_w_last_prices = pd.merge_asof(
            prediction_candidates.sort_index(),
            previous_observations.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            suffixes=("", "_previous")
        )

        # Spočíta delty: (ranní pozorování) - (průměr přechozí den odpoledne)
        for col in self.spec.predictor_cols:
            miss_w_last_prices[f"{col}_delta"] = (
                    miss_w_last_prices[col] - miss_w_last_prices[f"{col}_previous"]
            )

        # Musíš vybrat správné názvy prediktorů (model je natrénován na M_1,Y_1, zde máš navíc _delta) -> úprava mappingu
        exog_names = model.model.exog_names
        non_const = [name for name in exog_names if name != "const"]
        feature_map = {}
        for name in non_const:
            feature_map[name] = f"{name}_delta"

        # poskládej X ve správném pořadí a přejmenuj sloupce v X na názvy podle modelu
        X = miss_w_last_prices[[feature_map[name] for name in non_const]].copy()
        X.columns = non_const

        # Predikuj delta
        X = sm.add_constant(X, has_constant="add")
        miss_w_last_prices["delta_prediction"] = model.predict(X)

        # Spočti výslednou predikci ceny
        miss_w_last_prices[f"{self.spec.target_col}_prediction"] = (
                miss_w_last_prices[f"{self.spec.target_col}_previous"] +
                miss_w_last_prices["delta_prediction"]
        )
        prediction = miss_w_last_prices[f"{self.spec.target_col}_prediction"]
        return prediction


    def predict_whole_month(self, data: pd.DataFrame, data_w_contract_id: pd.DataFrame, prediction_month: str) -> pd.DataFrame:
        """
        Zkombinuje predikce v konkrétním měsíci -> predikce první den + predikce po zbytek měsíce
        Pro OOS predicke vlož data z celého období (musíš znát přechod měsíce in sample -> out of sample)

        :param data: minutová data s cenami
        :param data_w_contract_id: minutová data contract_id, product_id, price
        :param prediction_month:
        :return: Predicke targetu pro celý měsíc
        """
        previous_obs_regular = self.inputs.previous_regular(data)
        candidates_regular = self.inputs.candidates_regular(data, prediction_month)
        prediction_regular = self.predict_from_inputs(candidates_regular, previous_obs_regular)

        try:
            candidates_first_day = self.inputs.candidates_first_day(data_w_contract_id, prediction_month)
            previous_obs_first_day = self.inputs.previous_first_day(data_w_contract_id, candidates_first_day)
            prediction_first_day = self.predict_from_inputs(candidates_first_day, previous_obs_first_day)

        except KeyError:
            print(f"No prediction for {self.spec.target_col} in month {prediction_month}")
            prediction_first_day = pd.Series(dtype=float)


        if prediction_first_day.empty:
            df_final = prediction_regular.to_frame(name=f"{self.spec.target_col}_prediction")
            df_final.index.name = "index"
            return df_final

        prediction_all = pd.concat([prediction_first_day, prediction_regular], axis=0).sort_index()
        df_final = prediction_all.to_frame(name=f"{self.spec.target_col}_prediction")
        df_final.index.name = "index"

        return df_final



#SPUŠTENÍ je v: create_predictions.ipynb