import csv
import io
import json


class DeliveryAgent:
    def to_csv(self, dataframes):
        result = {}
        for name, df in dataframes.items():
            if df.empty:
                continue
            buf = io.BytesIO()
            df.to_csv(buf, index=False)
            buf.seek(0)
            result[name] = buf.getvalue()
        return result

    def to_json(self, dataframes):
        result = {}
        for name, df in dataframes.items():
            if df.empty:
                continue
            if isinstance(df, dict):
                payload = df
            else:
                payload = json.loads(df.to_json(orient="records", date_format="iso"))
            result[name] = json.dumps(payload, indent=2).encode("utf-8")
        return result
