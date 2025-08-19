from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.operators.python import PythonOperator
from google.cloud import bigquery, storage
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import logging
import io

# Configuration
PROJECT_ID = 'mydevproject-468021'
DATASET_ID = 'bicycle_hire'
TABLE_ID = 'training_data'
BUCKET_NAME = 'us-central1-my-composer-env-ffd3281c-bucket'

# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'ml_pipeline_dag_v3_eu_mock',
    default_args=default_args,
    description='ML Pipeline for London Bicycle Trip Duration Prediction',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday
    catchup=False,
    tags=['ml', 'bigquery', 'gcs', 'bicycle-data'],
)

# SQL query for data extraction
extract_query = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` AS
WITH mock AS (
  SELECT 600 AS duration, TIMESTAMP('2024-01-01 08:15:00+00') AS start_date, 101 AS start_station_id, 205 AS end_station_id UNION ALL
  SELECT 300, TIMESTAMP('2024-01-02 17:45:00+00'), 102, 210 UNION ALL
  SELECT 900, TIMESTAMP('2024-02-10 12:30:00+00'), 103, 220 UNION ALL
  SELECT 120, TIMESTAMP('2024-03-05 06:10:00+00'), 104, 230
)
SELECT 
    duration,
    EXTRACT(HOUR FROM start_date) as start_hour,
    EXTRACT(DAYOFWEEK FROM start_date) as day_of_week,
    EXTRACT(MONTH FROM start_date) as month,
    start_station_id,
    end_station_id,
    COUNT(*) OVER (PARTITION BY start_station_id) as start_station_popularity,
    COUNT(*) OVER (PARTITION BY end_station_id) as end_station_popularity,
    ABS(start_station_id - end_station_id) as station_distance_proxy
FROM mock
LIMIT 10000
"""

def train_model(**context):
    """
    Train a linear regression model to predict trip duration
    """
    try:
        logging.info("Starting model training...")
        
        # Initialize BigQuery client
        client = bigquery.Client(project=PROJECT_ID)
        
        # Load data from BigQuery table
        query = f"""
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        """
        
        logging.info("Loading data from BigQuery...")
        # Run the query in EU; pass location directly to client.query
        df = client.query(query, location='EU').to_dataframe()
        logging.info(f"Loaded {len(df)} rows of data")
        
        # Prepare features and target
        feature_columns = ['start_hour', 'day_of_week', 'month', 'start_station_id', 
                          'end_station_id', 'start_station_popularity', 
                          'end_station_popularity', 'station_distance_proxy']
        
        X = df[feature_columns]
        y = df['duration']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'r2_score': float(r2),
            'mean_squared_error': float(mse),
            'mean_absolute_error': float(mae),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        logging.info(f"Model metrics: R² = {r2:.4f}, MSE = {mse:.2f}, MAE = {mae:.2f}")
        
        # Initialize GCS client
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Save model to GCS
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        
        model_blob = bucket.blob(f'models/linear_regression_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
        model_blob.upload_from_file(model_buffer, content_type='application/octet-stream')
        
        logging.info(f"Model saved to gs://{BUCKET_NAME}/models/")
        
        # Save metrics to GCS
        metrics_json = json.dumps(metrics, indent=2)
        metrics_blob = bucket.blob(f'metrics/metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        metrics_blob.upload_from_string(metrics_json, content_type='application/json')
        
        logging.info(f"Metrics saved to gs://{BUCKET_NAME}/metrics/")
        
        # Return metrics for downstream tasks
        return metrics
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

def log_completion(**context):
    """
    Log completion message with metrics summary
    """
    try:
        # Get metrics from upstream task
        task_instance = context['task_instance']
        metrics = task_instance.xcom_pull(task_ids='train_model_task')
        
        if metrics:
            logging.info("🎉 Model training complete! Check GCS for saved model and metrics")
            logging.info(f"📊 Model Performance Summary:")
            logging.info(f"   - R² Score: {metrics['r2_score']:.4f}")
            logging.info(f"   - Mean Squared Error: {metrics['mean_squared_error']:.2f}")
            logging.info(f"   - Mean Absolute Error: {metrics['mean_absolute_error']:.2f}")
            logging.info(f"   - Training Samples: {metrics['training_samples']}")
            logging.info(f"   - Test Samples: {metrics['test_samples']}")
            logging.info(f"📦 Artifacts saved to gs://{BUCKET_NAME}/")
        else:
            logging.warning("No metrics received from training task")
            
    except Exception as e:
        logging.error(f"Error in completion logging: {str(e)}")
        # Don't raise exception here to avoid failing the entire pipeline

# Task 1: Extract data from BigQuery
extract_data_task = BigQueryInsertJobOperator(
    task_id='extract_data_task',
    configuration={
        'query': {
            'query': extract_query,
            'useLegacySql': False,
        }
    },
    location='EU',
    dag=dag,
)

# Task 2: Train ML model
train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag,
)

# Task 3: Log completion
log_completion_task = PythonOperator(
    task_id='log_completion_task',
    python_callable=log_completion,
    dag=dag,
)

# Set task dependencies
extract_data_task >> train_model_task >> log_completion_task