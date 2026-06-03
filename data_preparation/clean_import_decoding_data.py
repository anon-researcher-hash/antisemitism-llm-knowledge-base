from utils.data_helpers import remove_author
from utils.data_helpers import strip_anchor_tags
from utils.db_helpers import execute_sql_select, create_postgres_engine, write_to_db

# Decoding data had already author tags removed and URLs replaced by REMOVED_URL
df = execute_sql_select("""SELECT * FROM decoding_old""")  # previously renamed the table form decoding to decoding_old
df["comment_cleaned"] = df["comment_replaced_all"].map(lambda x: x.strip()).map(
    strip_anchor_tags).map(lambda x: x.strip()).map(lambda x: x.replace("\u200b", "")).map(
    lambda x: x.replace("REMOVED_REPLYTO", "")).map(lambda x: x.replace("REMOVED_URL", "")).map(remove_author)

con = create_postgres_engine()
write_to_db(df=df, table_name="decoding", con=con, if_exists="replace")
