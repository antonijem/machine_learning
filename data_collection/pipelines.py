from itemadapter import ItemAdapter
import mysql.connector
import logging

class PszprojekatPipeline(object):

    def __init__(self):
        self.create_connection()
        self.create_table()

    def create_connection(self):
        try:
            logging.debug("Connecting to the database...")
            self.conn = mysql.connector.connect(
                host='localhost',
                user='root',
                passwd='1234',
                database='psz'
            )
            self.curr = self.conn.cursor()
            logging.debug("Database connection successful.")
        except mysql.connector.Error as err:
            logging.error(f"Error connecting to database: {err}")
            raise

    def create_table(self):
        try:
            logging.debug("Creating table...")
            self.curr.execute("""DROP TABLE IF EXISTS books_tb""")
            self.curr.execute("""create table books_tb(
                title TEXT,
                price INT,
                author TEXT,
                category TEXT,
                publisher TEXT,
                year TEXT,
                pages INT,
                cover TEXT,
                format TEXT,
                description TEXT
            )""")
            logging.debug("Table created successfully.")
        except mysql.connector.Error as err:
            logging.error(f"Error creating table: {err}")
            raise

    def process_item(self, item, spider):
        try:
            logging.debug(f"Processing item: {item}")
            self.store_db(item)
        except mysql.connector.Error as err:
            logging.error(f"Error processing item: {err}")
        return item

    def store_db(self, item):
        # Handle potential None values and ensure proper extraction
        title = item.get('title', None)
        author = item.get('author', None)
        category = item.get('category', None)
        publisher = item.get('publisher', None)
        year = item.get('year', None)
        cover = item.get('cover', None)
        format_ = item.get('format', None)
        description = item.get('description', None)

        try:
            raw_price = item['price'].split(',')[0]
            price = int(raw_price.replace('.', '').strip())
            pages = int(item['pages'].strip())
        except ValueError as e:
            logging.error(f"Error converting price or pages to int: {e}")
            price = 0
            pages = 0

        try:
            logging.debug(f"Inserting item into database: {item}")
            self.curr.execute("""INSERT INTO books_tb (title, price, author, category, publisher, year, pages, cover, format, description) 
                                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", (
                title,
                price,
                author,
                category,
                publisher,
                year,
                pages,
                cover,
                format_,
                description
            ))
            self.conn.commit()
            logging.debug("Item inserted successfully.")
        except mysql.connector.Error as err:
            logging.error(f"Error inserting item into database: {err}")
            raise

# Enable logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs, INFO or ERROR for less verbosity
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
