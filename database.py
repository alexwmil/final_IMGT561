import sqlite3
import dataclasses
 
@dataclasses.dataclass
class Stock:
    rowid: int
    name: str
    date: str
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class Database:

    def __init__(self, database_file_path:str):
        self.database_file_path = database_file_path
        self._init_database_schema()
    
    def _init_database_schema(self):
        with sqlite3.connect(self.database_file_path) as conn:
            # create a cursor object
            cur = conn.cursor()
            
            cur.execute('''
                    CREATE TABLE IF NOT EXISTS stock (
                        rowid INTEGER PRIMARY KEY,
                        name STRING,
                        date STRING,
                        time STRING NOT NULL,
                        open FLOAT NOT NULL,
                        high FLOAT NOT NULL,
                        low FLOAT NOT NULL,
                        close FLOAT NOT NULL,
                        volume INTEGER NOT NULL

                    )
                ''')
            conn.commit()
            

    def insert(self,stock):
        with sqlite3.connect(self.database_file_path) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO stock (rowid, name, date, time, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?,?)", (stock.rowid, stock.name, stock.date, stock.time, stock.open, stock.high, stock.low, stock.close, stock.volume))

            conn.commit()
            
            return cur.lastrowid

    def add_indexes_to_database(self):

        with sqlite3.connect(self.database_file_path) as conn:
            # TODO add logic to create or more column indexes to the
            # created table
            cur = conn.cursor()
            conn.execute('''
                CREATE INDEX idx_date ON stock (
                    date
                )
            ''')
            conn.commit()

        pass

    def retrieve_single(self, id):
        with sqlite3.connect(self.database_file_path) as conn:
            cur = conn.cursor()
            
            row = cur.execute('''
                SELECT * FROM stock WHERE rowid = ?
            ''', (id,)).fetchone()
            conn.commit()
            
            row = Stock(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8])
            return row
            
            




