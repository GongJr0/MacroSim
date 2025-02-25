import sqlite3 as sql
from typing import Any

class BaseCache:
    def __init__(self, file: str = 'cache.db', init_command: str = ...) -> None:

        self.file = file
        self.con, self.cur = self.connect()

        self.exec(init_command)

    def connect(self) -> tuple[sql.Connection, sql.Cursor]:
        con = sql.connect(self.file)
        cur = con.cursor()

        return con, cur

    def exec(self, command: str, var: tuple[Any]) -> None:
        self.cur.execute(command, var)
        self.con.commit()
