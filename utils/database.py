"""
Database utilities for PyTorch RLM Analysis
Handles PostgreSQL connections and common operations
"""

import pandas as pd
from sqlalchemy import create_engine, text
from config import DATABASE_URL


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, db_url=None):
        self.db_url = db_url or DATABASE_URL
        self.engine = create_engine(self.db_url)

    def save_dataframe(self, df, table_name, if_exists="replace"):
        """Save pandas DataFrame to database"""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        print(f"✓ Saved {len(df)} rows to table '{table_name}'")

    def load_dataframe(self, table_name, query=None):
        """Load data from database into pandas DataFrame"""
        if query:
            return pd.read_sql(query, self.engine)
        return pd.read_sql(f"SELECT * FROM {table_name}", self.engine)

    def table_exists(self, table_name):
        """Check if table exists"""
        query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table
            );
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"table": table_name})
            return result.scalar()

    def get_table_stats(self, table_name):
        """Get row count and column info for a table"""
        count_query = text(f"SELECT COUNT(*) FROM {table_name}")
        with self.engine.connect() as conn:
            count = conn.execute(count_query).scalar()

        columns_query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = :table
            ORDER BY ordinal_position;
        """)
        with self.engine.connect() as conn:
            columns = pd.read_sql(columns_query, conn, params={"table": table_name})

        return {
            "table": table_name,
            "row_count": count,
            "columns": columns.to_dict("records"),
        }

    def list_tables(self):
        """List all tables in the database"""
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn)
            return result["table_name"].tolist()

    def execute_query(self, query, params=None):
        """Execute raw SQL query"""
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result

    def create_indexes(self, table_name, columns):
        """Create indexes on specified columns"""
        for col in columns:
            index_name = f"idx_{table_name}_{col}"
            query = text(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({col});"
            )
            with self.engine.connect() as conn:
                conn.execute(query)
                conn.commit()
            print(f"✓ Created index on {table_name}.{col}")


def get_db_manager():
    """Factory function to get database manager instance"""
    return DatabaseManager()


if __name__ == "__main__":
    # Test database connection
    db = get_db_manager()
    tables = db.list_tables()
    print(f"Connected to database. Tables: {tables}")
