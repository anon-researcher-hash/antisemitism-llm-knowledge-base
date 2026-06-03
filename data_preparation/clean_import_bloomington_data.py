import ftfy
import pandas as pd

from config import *
from utils.data_helpers import remove_author, remove_url, remove_rt, \
    fix_encoding_of_apostrophes
from utils.db_helpers import write_to_db, create_postgres_engine

if __name__ == '__main__':

    # read original file with corrupt encoding
    import_file_path = join(PROJECT_DIR, DATA_DIR, "EN_gold_combined.csv")
    df = pd.read_csv(import_file_path, encoding="utf-8")

    # rename columns to fit postgres naming conventions
    df.columns = ['id', 'username', 'create_date', 'code', 'keyword', 'text',
                  'ihra_section_1', 'ihra_section_2', 'keyword_2']

    # dataset contained keyword column twice, therefore we drop the second one
    df.drop(columns=['keyword_2'], inplace=True)
    print(df.columns)

    for col in df.columns:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(x) if isinstance(x, str) else x)

    df["comment_cleaned"] = df["text"].map(fix_encoding_of_apostrophes).map(remove_rt).map(remove_author).map(
        remove_url)

    # save in correct encoding
    export_file_path = join(PROJECT_DIR, DATA_DIR, "EN_gold_combined_fixed.csv")
    df.to_csv(export_file_path, index=False, encoding="utf-8")

    con = create_postgres_engine()
    write_to_db(df=df, table_name="bloomington", con=con, if_exists="replace")
