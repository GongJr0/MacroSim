from BaseCache import BaseCache
import pandas as pd

class Logger:
    def __init__(self):
        self.logger = BaseCache(
            file='logs.db',
            init_command="""CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, 
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, message TEXT)
            """
        )

    def log(self, message):
        command = "INSERT INTO logs (message,) VALUES (?,)"
        self.logger.exec(command, (message,))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_sql(sql='logs',
                           con=self.logger.con,
                           index_col='id',
                           parse_dates=['timestamp'])
