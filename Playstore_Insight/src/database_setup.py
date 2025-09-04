import sqlite3
import os
from datetime import datetime, timedelta

class DatabaseManager:
    def __init__(self, db_path="trend_analysis.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id TEXT UNIQUE,
                app_name TEXT,
                content TEXT,
                rating INTEGER,
                review_date DATE,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes separately
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_review_date ON reviews(review_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_app_name ON reviews(app_name)')
        
        # Topics table with seed topics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_name TEXT UNIQUE,
                is_seed_topic BOOLEAN DEFAULT FALSE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parent_topic_id INTEGER,
                confidence_score REAL DEFAULT 0.0,
                FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
            )
        ''')
        
        # Review-topic mappings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS review_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                topic_id INTEGER,
                confidence_score REAL,
                assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (review_id) REFERENCES reviews(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                UNIQUE(review_id, topic_id)
            )
        ''')
        
        # Daily trends table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                trend_date DATE,
                frequency_count INTEGER DEFAULT 0,
                sentiment_avg REAL DEFAULT 0.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                UNIQUE(topic_id, trend_date)
            )
        ''')
        
        # Insert seed topics for food delivery apps
        seed_topics = [
            "delivery_delay", "food_quality", "delivery_partner_behavior", 
            "app_performance", "payment_issues", "order_cancellation",
            "customer_service", "pricing_concerns", "packaging_issues",
            "restaurant_quality", "tracking_issues", "refund_problems"
        ]
        
        for topic in seed_topics:
            cursor.execute('''
                INSERT OR IGNORE INTO topics (topic_name, is_seed_topic) 
                VALUES (?, TRUE)
            ''', (topic,))
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def insert_reviews(self, reviews_data):
        """Batch insert reviews"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for review in reviews_data:
            cursor.execute('''
                INSERT OR IGNORE INTO reviews 
                (review_id, app_name, content, rating, review_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (review['review_id'], review['app_name'], review['content'], 
                  review['rating'], review['review_date']))
        
        conn.commit()
        conn.close()
        return len(reviews_data)

if __name__ == "__main__":
    db = DatabaseManager()
    print("Database setup completed!")