import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

from config import TEXT_COL, TABLE_BLO, TABLE_DEC

load_dotenv()

HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
USER = os.getenv("DB_USER")
PW = os.getenv("DB_PASSWORD")
DATABASE = os.getenv("DB_NAME")


def execute_sql_select(command, return_result_as_df=True):
    """Per default returns the result as a pandas DataFrame"""
    conn = None
    try:
        conn = psycopg2.connect(
            database=DATABASE,
            host=HOST,
            port=PORT,
            user=USER,
            password=PW,
            sslmode="require",
        )

        cur = conn.cursor()
        cur.execute(command)
        colnames = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        cur.close()
        conn.commit()
        if return_result_as_df is True:
            return pd.DataFrame.from_records(data, columns=colnames)
        else:
            return colnames, data
    finally:
        if conn is not None:
            close_connection(conn)


def create_postgres_engine(
        database=DATABASE, host=HOST, port=PORT, user=USER, password=PW
):
    db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url, client_encoding="utf8")

    return engine


def write_to_db(df, table_name, con, dtype=None, index=True, if_exists="fail",
                # if_exists options are: "fail", "replace", "append"
                ):
    df.to_sql(
        table_name,
        con=con,
        dtype=dtype,
        if_exists=if_exists,
        index=index,
    )


def close_connection(conn):
    conn.close()
    print("Connection to DB closed")


def fetch_data(source: str, label: str) -> pd.DataFrame:
    """Fetch all data from DB with additional useful attributes depending on the source and label.
    source: either 'bloomington' or 'decoding'
    """
    if source == "bloomington":
        sql_statement = f"""SELECT {TEXT_COL}, keyword, ihra_section_1, ihra_section_2 FROM {TABLE_BLO} WHERE code = {label}"""
    elif source == "decoding":
        label = "I" + label
        sql_statement = f"""SELECT {TEXT_COL}, discourse, comment_level, comment_codes_all, source_outlet FROM {TABLE_DEC} WHERE code = '{label}' AND comment_codes_all != 'I1';"""
    else:
        print(f"SOURCE {source} not supported")
        return pd.DataFrame()
    print(sql_statement)
    data = execute_sql_select(sql_statement)
    return data.drop_duplicates()
