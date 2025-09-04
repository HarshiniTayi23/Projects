import requests
from google_play_scraper import reviews, Sort
import pandas as pd
from datetime import datetime, timedelta
import time
import random
from .database_setup import DatabaseManager

class DataIngestion:
    def __init__(self, app_id="com.swiggy.android", db_manager=None):
        self.app_id = app_id  # Swiggy app ID
        self.db_manager = db_manager or DatabaseManager()
        
    def scrape_reviews_batch(self, start_date, end_date, count=500):
        """
        Scrape reviews for a specific date range
        Returns reviews in batches to manage memory efficiently
        """
        try:
            print(f"Scraping reviews from {start_date} to {end_date}")
            
            # Get reviews from Google Play Store
            result, _ = reviews(
                self.app_id,
                lang='en',
                country='in',
                sort=Sort.NEWEST,
                count=count
            )
            
            # Filter reviews by date range
            filtered_reviews = []
            for review in result:
                review_date = review['at'].date()
                if start_date <= review_date <= end_date:
                    filtered_reviews.append({
                        'review_id': review['reviewId'],
                        'app_name': 'swiggy',
                        'content': review['content'] or '',
                        'rating': review['score'],
                        'review_date': review_date
                    })
            
            return filtered_reviews
            
        except Exception as e:
            print(f"Error scraping reviews: {e}")
            return []
    
    def process_daily_batch(self, target_date):
        """Process reviews for a specific date"""
        start_date = target_date
        end_date = target_date
        
        # Scrape reviews
        reviews_data = self.scrape_reviews_batch(start_date, end_date)
        
        if reviews_data:
            # Insert into database
            count = self.db_manager.insert_reviews(reviews_data)
            print(f"Inserted {count} reviews for {target_date}")
            return count
        else:
            print(f"No reviews found for {target_date}")
            return 0
    
    def simulate_historical_data(self, start_date, end_date):
        """
        Simulate historical data collection from June 2024 to current date
        In production, this would be replaced by actual daily scraping
        """
        current_date = start_date
        total_processed = 0
        
        while current_date <= end_date:
            print(f"Processing batch for {current_date}")
            
            # Get some sample reviews for simulation
            try:
                # In real implementation, this would scrape actual historical data
                # For demo, we'll get current reviews and simulate dates
                result, _ = reviews(
                    self.app_id,
                    lang='en',
                    country='in',
                    sort=Sort.NEWEST,
                    count=50  # Smaller batches for efficiency
                )
                
                # Simulate reviews for the current date
                simulated_reviews = []
                for i, review in enumerate(result[:random.randint(10, 30)]):
                    simulated_reviews.append({
                        'review_id': f"{review['reviewId']}_{current_date}_{i}",
                        'app_name': 'swiggy',
                        'content': review['content'] or f"Sample review for {current_date}",
                        'rating': review['score'],
                        'review_date': current_date
                    })
                
                if simulated_reviews:
                    count = self.db_manager.insert_reviews(simulated_reviews)
                    total_processed += count
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        print(f"Historical data simulation completed. Total reviews: {total_processed}")
        return total_processed

if __name__ == "__main__":
    # Test data ingestion
    ingestion = DataIngestion()
    
    # Simulate some historical data
    start_date = datetime(2024, 6, 1).date()
    end_date = datetime(2024, 6, 30).date()  
    
    ingestion.simulate_historical_data(start_date, end_date)
