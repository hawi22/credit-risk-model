from sklearn.cluster import KMeans
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom Transformer to extract date features and aggregates."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convert to datetime
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # 1. Extract Time Features
        X['Hour'] = X['TransactionStartTime'].dt.hour
        X['Day'] = X['TransactionStartTime'].dt.day
        X['Month'] = X['TransactionStartTime'].dt.month
        
        # 2. Basic Transaction Features
        
        X['Transaction_Type'] = np.where(X['Amount'] > 0, 'Debit', 'Credit')
        
        return X.drop(columns=['TransactionStartTime', 'TransactionId', 'BatchId', 
                               'AccountId', 'SubscriptionId', 'CustomerId'], errors='ignore')

def build_processing_pipeline(categorical_cols, numerical_cols):
    """Creates a full automation pipeline."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    full_pipeline = Pipeline(steps=[
        ('feature_extraction', FeatureExtractor()),
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    # 1. Load Raw Data
    df = pd.read_csv('data/raw/data.csv')
    
    # --- TASK 4: PROXY TARGET ENGINEERING ---
    print("Step 1: Calculating RFM metrics...")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max()
    
    # Calculate Recency, Frequency, Monetary
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency', 
        'TransactionId': 'Frequency', 
        'Amount': 'Monetary'
    })

    # Pre-process (Scale) RFM for Clustering
    scaler_rfm = StandardScaler()
    rfm_scaled = scaler_rfm.fit_transform(rfm)

    # Cluster Customers into 3 groups
    print("Step 2: Clustering customers...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Define High-Risk: Cluster with lowest average Monetary/Frequency
    # We analyze clusters to find the "least engaged"
    cluster_stats = rfm.groupby('Cluster').agg({'Monetary':'mean', 'Frequency':'mean'})
    bad_cluster = cluster_stats['Monetary'].idxmin()
    
    rfm['is_high_risk'] = (rfm['Cluster'] == bad_cluster).astype(int)
    print(f"High-risk label assigned to Cluster {bad_cluster}")

    # Integrate back to main dataframe
    df = df.merge(rfm[['is_high_risk']], on='CustomerId', how='left')
    
    # --- TASK 3: FEATURE PIPELINE ---
    print("Step 3: Running feature engineering pipeline...")
    num_features = ['Amount', 'Value']
    cat_features = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    
    pipeline = build_processing_pipeline(cat_features, num_features)
    X_processed = pipeline.fit_transform(df)
    
    # Create final dataframe
    final_df = pd.DataFrame(X_processed)
    final_df['is_high_risk'] = df['is_high_risk'].values
    
    # Save the deliverable
    os.makedirs('data/processed', exist_ok=True)
    final_df.to_csv('data/processed/processed_data.csv', index=False)
    
    print("\nDeliverable Complete: data/processed/processed_data.csv")
    print(f"Target Distribution:\n{final_df['is_high_risk'].value_counts()}")

def create_proxy_target(df):
    """Calculates RFM and assigns a risk label using K-Means."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Reference date for Recency
    snapshot_date = df['TransactionStartTime'].max()
    
    # Aggregate by Customer
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days, # Recency
        'TransactionId': 'count',                                         # Frequency
        'Amount': 'sum'                                                  # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })

    # Cluster to identify High Risk
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(rfm))
    
    # Identify the 'Bad' cluster (Usually low frequency, low monetary, high recency)
    bad_cluster = rfm.groupby('Cluster')['Monetary'].mean().idxmin()
    rfm['is_high_risk'] = (rfm['Cluster'] == bad_cluster).astype(int)
    
    return rfm[['is_high_risk']]