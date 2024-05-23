import logging
import time
import sys
from tqdm import tqdm
import os
from contextlib import redirect_stdout
import io

# Ensure the utils module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.StockHistoryFetcher import StockHistoryFetcher  # Changed to absolute import

# Setup logging
log_filename = f"logs/history_fetcher_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Create an instance of StockHistoryFetcher
fetcher = StockHistoryFetcher(None)

try:
    # Get all stock codes
    all_stock_codes_df = fetcher.fetch_all_stock_codes()
    
    # Loop through all stock codes and fetch historical data for each with a progress bar
    with tqdm(total=all_stock_codes_df.shape[0], desc="Fetching History") as pbar:
        for index, row in all_stock_codes_df.iterrows():
            stock_code = row['code']
            fetcher.stock_code = stock_code  # Update the stock code in fetcher
            logging.info(f"Fetching history for stock code: {stock_code}")
            
            # Update tqdm description with current stock code
            pbar.set_description(f"Fetching history for stock code: {stock_code}")
            # Redirect fetch_history output to log file
            log_stream = io.StringIO()
            with redirect_stdout(log_stream):
                fetcher.fetch_history()
            logging.info(log_stream.getvalue())
            
            # Update progress bar
            pbar.update(1)
finally:
    # Ensure resources are released after fetching is done
    logging.info("Completed fetching all stock histories.")
    del fetcher

logging.shutdown()
