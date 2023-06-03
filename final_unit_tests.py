import database
import sqlite3
import unittest
import os
import tempfile

def count_tables(db_file_path:str):

    with sqlite3.connect(db_file_path) as conn:
        count = conn.execute('''
        SELECT 
            COUNT(*)
        FROM 
            sqlite_schema
        WHERE 
            type ='table' AND 
            name NOT LIKE 'sqlite_%';
        ''').fetchone()[0]
    return count

def count_indexes(db_file_path:str):

    with sqlite3.connect(db_file_path) as conn:
        count = conn.execute('''
        SELECT 
            COUNT(*)
        FROM 
            sqlite_schema
        WHERE 
            type ='index' AND 
            name NOT LIKE 'sqlite_%';
        ''').fetchone()[0]
    return count


class TestDatabaseFunctions(unittest.TestCase):

    def test_db_create_table(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            db_file_path = os.path.join(tempdirname, "data.db")
            
            before_count = count_tables(db_file_path)

            database.Database(db_file_path)

            after_count = count_tables(db_file_path)
        
            os.remove(db_file_path)
        print(f"Table count before {before_count} and after {after_count}")
        self.assertGreater(after_count, before_count)

    def test_db_create_index(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            db_file_path = os.path.join(tempdirname, "data.db")
            
            before_count = count_indexes(db_file_path)

            ds = database.Database(db_file_path)
            ds.add_indexes_to_database()

            after_count = count_indexes(db_file_path)
        
            os.remove(db_file_path)
        print(f"Index count before {before_count} and after {after_count}")
        self.assertGreater(after_count, before_count)

    def test_db_insert(self):
        with tempfile.TemporaryDirectory() as tempdirname:
            db_file_path = os.path.join(tempdirname, "data.db")
            rowid = 1
            test_data = database.Stock(rowid,"IBM","2023-5-21","1:48",100,100,100,100,1)
            ds = database.Database(db_file_path)
            ds.insert(test_data)
            retrieved_data = ds.retrieve_single(rowid)

            self.assertEqual(test_data,retrieved_data)

if __name__ == "__main__":
    unittest.main()