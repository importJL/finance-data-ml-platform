import duckdb
import pandas as pd


class PITFeatureStore:
    def __init__(self):
        self._con = duckdb.connect(":memory:")
        self._registered = set()

    def ingest(self, name, df, date_col="Date"):
        if df is None or df.empty:
            return
        if date_col not in df.columns:
            df = df.copy()
            df[date_col] = pd.NaT
        self._con.register(f"_{name}", df)
        self._registered.add(name)

    def _ensure_date_indexed(self, name, date_col="Date"):
        if name not in self._registered:
            return False
        return True

    def as_of_join(self, target_df, feature_names, date_col="Date"):
        if target_df is None or target_df.empty:
            return target_df

        self._con.register("_target", target_df)

        result = "_target"
        for feat_name in feature_names:
            if feat_name not in self._registered:
                continue
            alias = f"_{feat_name}"
            result = (
                f"SELECT t.*, f.* EXCLUDE({date_col}) "
                f"FROM {result} t "
                f"LEFT JOIN {alias} f "
                f"ON f.{date_col} <= t.{date_col} "
                f"QUALIFY ROW_NUMBER() OVER ("
                f"  PARTITION BY t.{date_col} "
                f"  ORDER BY f.{date_col} DESC"
                f") = 1"
            )

        return self._con.execute(f"SELECT * FROM ({result}) t ORDER BY t.{date_col}").df()

    def forward_fill(self, df, full_dates, date_col="Date", fill_cols=None):
        if df is None or df.empty:
            return df

        full = pd.DataFrame({date_col: full_dates})
        merged = full.merge(df, on=date_col, how="left")

        if fill_cols is None:
            fill_cols = [c for c in df.columns if c != date_col]

        for col in fill_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill()

        return merged

    def build_feature_matrix(self, target_df, feature_map, date_col="Date"):
        aligned = target_df.copy()
        for feat_name, feat_df in feature_map.items():
            if feat_df is None or feat_df.empty:
                continue
            feat_name_clean = feat_name.replace(" ", "_").replace("-", "_")
            self.ingest(feat_name_clean, feat_df, date_col)
            aligned = self.as_of_join(aligned, [feat_name_clean], date_col)
        return aligned

    def clear(self):
        self._con.close()
        self._con = duckdb.connect(":memory:")
        self._registered = set()

    def leakage_check(self, feature_df, target_col):
        warnings = []
        for col in feature_df.columns:
            if col in ["Date", target_col]:
                continue
            num_valid = feature_df[col].notna().sum()
            if num_valid < 10:
                continue
            match_mask = feature_df[col].notna() & feature_df[target_col].notna()
            if match_mask.sum() < 10:
                continue
            corr = feature_df.loc[match_mask, col].corr(feature_df.loc[match_mask, target_col])
            if abs(corr) > 0.05:
                warnings.append(f"Leakage suspect: {col} has contemporaneous correlation {corr:.4f} with {target_col}")
        return warnings
