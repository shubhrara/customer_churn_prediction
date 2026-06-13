import os
import pandas as pd
import urllib.request

def download_data():
    raw_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    dest_dir = os.path.join("data", "raw")
    dest_path = os.path.join(dest_dir, "Telco-Customer-Churn.csv")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading raw Telco Customer Churn data from: {raw_url}")
    try:
        urllib.request.urlretrieve(raw_url, dest_path)
        print(f"Dataset successfully downloaded and saved to: {dest_path}")
        
        # Verify file loading
        df = pd.read_csv(dest_path)
        print(f"Verification successful: Loaded dataset with shape {df.shape}")
        return dest_path
    except Exception as e:
        print(f"Error during dataset download: {e}")
        raise e

if __name__ == "__main__":
    download_data()
