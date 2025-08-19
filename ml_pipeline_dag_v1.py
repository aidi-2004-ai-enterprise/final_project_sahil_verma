from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import bigquery, storage
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'ml_pipeline_london_bikes',
    default_args=default_args,
    description='ML Pipeline for London Bike Share Trip Duration Prediction',
    schedule_interval='0 0 * * 0',  # Weekly on Sunday
    catchup=False,
    tags=['ml', 'bigquery', 'london-bikes'],
)

# Task 1: Data Extraction (Cross-region with Python operator)
def extract_bike_data(**context):
    """Extract bike data from EU public dataset and save to US dataset"""
    
    from google.cloud import bigquery
    
    client = bigquery.Client()
    
    logging.info("🚀 Starting cross-region data extraction...")
    logging.info("📍 Source: EU public dataset (bigquery-public-data.london_bicycles.cycle_hire)")
    logging.info("📍 Destination: US dataset (mydevproject-468021.ml_pipeline.bike_features)")
    
    # Step 1: Query data from EU public dataset
    query = """
    SELECT 
        duration,
        EXTRACT(HOUR FROM start_date) as start_hour,
        EXTRACT(DAYOFWEEK FROM start_date) as day_of_week,
        EXTRACT(MONTH FROM start_date) as month,
        start_station_id,
        end_station_id,
        -- Create features from station popularity
        COUNT(*) OVER (PARTITION BY start_station_id) as start_station_popularity,
        COUNT(*) OVER (PARTITION BY end_station_id) as end_station_popularity,
        -- Distance proxy (difference in station IDs as rough estimate)
        ABS(start_station_id - end_station_id) as station_distance_proxy
    FROM 
        `bigquery-public-data.london_bicycles.cycle_hire`
    WHERE 
        duration IS NOT NULL 
        AND duration > 0 
        AND duration < 86400  -- Less than 24 hours (remove outliers)
        AND start_date >= '2022-01-01'  -- Recent data only
        AND start_date < '2023-01-01'
    LIMIT 10000  -- Keep dataset manageable
    """
    
    logging.info("📊 Querying data from EU public dataset...")
    
    try:
        # Execute query in EU region (location passed to query method)
        query_job = client.query(query, location="EU")
        df = query_job.to_dataframe()
        
        logging.info(f"✅ Successfully retrieved {len(df)} records from EU dataset")
        logging.info(f"📋 Data shape: {df.shape}")
        logging.info(f"📋 Columns: {list(df.columns)}")
        
        # Validate data quality
        if len(df) == 0:
            raise ValueError("No data retrieved from the source dataset")
        
        # Log sample statistics
        logging.info(f"📊 Duration statistics:")
        logging.info(f"   - Mean: {df['duration'].mean():.2f} seconds")
        logging.info(f"   - Min: {df['duration'].min()} seconds")
        logging.info(f"   - Max: {df['duration'].max()} seconds")
        
    except Exception as e:
        logging.error(f"❌ Failed to query EU dataset: {str(e)}")
        raise
    
    # Step 2: Upload data to US dataset
    logging.info("📤 Uploading data to US dataset...")
    
    # Define destination table in US region
    table_id = "mydevproject-468021.ml_pipeline.bike_features"
    
    try:
        # Configure load job
        job_config = bigquery.LoadJobConfig()
        job_config.write_disposition = "WRITE_TRUNCATE"  # Replace existing data
        job_config.create_disposition = "CREATE_IF_NEEDED"  # Create table if needed
        job_config.autodetect = True  # Auto-detect schema
        
        # Load dataframe to BigQuery table
        # This automatically uses the US region since the dataset is in US
        load_job = client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        
        # Wait for the load job to complete
        load_job.result()
        
        # Get the loaded table info
        table = client.get_table(table_id)
        
        logging.info(f"✅ Data successfully loaded to {table_id}")
        logging.info(f"📊 Final table info:")
        logging.info(f"   - Rows: {table.num_rows:,}")
        logging.info(f"   - Size: {table.num_bytes:,} bytes")
        logging.info(f"   - Location: {table.location}")
        logging.info(f"   - Created: {table.created}")
        
    except Exception as e:
        logging.error(f"❌ Failed to upload to US dataset: {str(e)}")
        raise
    
    # Return extraction results for downstream tasks
    extraction_results = {
        'rows_processed': len(df),
        'table_id': table_id,
        'table_location': table.location,
        'table_size_bytes': table.num_bytes,
        'timestamp': datetime.now().isoformat(),
        'duration_stats': {
            'mean': float(df['duration'].mean()),
            'min': int(df['duration'].min()),
            'max': int(df['duration'].max()),
            'std': float(df['duration'].std())
        }
    }
    
    logging.info("🎉 Data extraction completed successfully!")
    return extraction_results

extract_data_task = PythonOperator(
    task_id='extract_bike_data',
    python_callable=extract_bike_data,
    dag=dag,
)

# Task 2: Model Training & Persistence
def train_model(**context):
    """Train a model to predict bike trip duration and save to GCS"""
    
    # Initialize clients
    bq_client = bigquery.Client()
    storage_client = storage.Client()
    
    logging.info("🚀 Starting model training phase...")
    
    # Get extraction results from previous task
    extraction_results = context['ti'].xcom_pull(task_ids='extract_bike_data')
    logging.info(f"📊 Processing {extraction_results['rows_processed']:,} records")
    logging.info(f"🗂️  Source table: {extraction_results['table_id']}")
    
    # Determine GCS bucket name - Get the actual Composer bucket
    logging.info("🔍 Detecting GCS bucket...")
    
    # Try to get bucket from Airflow configuration
    dag_bucket = None
    try:
        from airflow.configuration import conf
        dag_bucket = conf.get('core', 'dags_folder', fallback='')
        logging.info(f"📁 DAGs folder: {dag_bucket}")
    except:
        logging.info("⚠️  Could not read Airflow config")
    
    # Extract bucket name if found in config
    if dag_bucket and 'gs://' in dag_bucket:
        bucket_name = dag_bucket.split('/')[2]
        logging.info(f"📦 Extracted bucket from config: {bucket_name}")
    else:
        # Try to discover buckets by listing available ones
        project_id = 'mydevproject-468021'
        logging.info(f"🔍 Searching for buckets in project: {project_id}")
        
        try:
            # List all buckets in the project
            buckets = list(storage_client.list_buckets())
            logging.info(f"📦 Found {len(buckets)} buckets in project")
            
            # Look for Composer-related buckets
            composer_buckets = [
                bucket.name for bucket in buckets 
                if any(keyword in bucket.name.lower() for keyword in ['composer', 'airflow', 'us-central1'])
            ]
            
            if composer_buckets:
                bucket_name = composer_buckets[0]  # Use the first Composer bucket found
                logging.info(f"✅ Found Composer bucket: {bucket_name}")
            else:
                # Fallback to first available bucket
                bucket_name = buckets[0].name if buckets else f"{project_id}-ml-models"
                logging.info(f"📦 Using fallback bucket: {bucket_name}")
                
        except Exception as e:
            # Final fallback
            bucket_name = f"{project_id}-ml-models"
            logging.warning(f"⚠️  Could not list buckets ({str(e)}), using fallback: {bucket_name}")
    
    # Try to access the bucket with error handling
    bucket = None
    try:
        bucket = storage_client.bucket(bucket_name)
        # Test access by trying to list one blob
        list(bucket.list_blobs(max_results=1))
        logging.info(f"✅ Successfully connected to GCS bucket: {bucket_name}")
        
    except Exception as e:
        logging.warning(f"⚠️  Could not access bucket {bucket_name}: {str(e)}")
        
        # Try to create the bucket if it doesn't exist
        try:
            bucket = storage_client.create_bucket(bucket_name, location="US-CENTRAL1")
            logging.info(f"✅ Created new GCS bucket: {bucket_name}")
        except Exception as create_error:
            logging.error(f"❌ Could not create bucket: {str(create_error)}")
            # Continue without GCS - model training will still work
            bucket = None
    
    # Load data from BigQuery (now in US region)
    logging.info("📥 Loading data from BigQuery...")
    
    query = """
    SELECT * FROM `mydevproject-468021.ml_pipeline.bike_features`
    """
    
    try:
        # Query from US region (where data now resides)
        query_job = bq_client.query(query, location="US")
        df = query_job.to_dataframe()
        
        logging.info(f"📊 Loaded {len(df):,} records from BigQuery")
        
        if len(df) == 0:
            raise ValueError("No data found in the target table")
            
    except Exception as e:
        logging.error(f"❌ Failed to load data from BigQuery: {str(e)}")
        raise
    
    # Prepare features and target
    logging.info("🔧 Preparing features and target variables...")
    
    feature_columns = [
        'start_hour', 'day_of_week', 'month', 'start_station_id', 
        'end_station_id', 'start_station_popularity', 
        'end_station_popularity', 'station_distance_proxy'
    ]
    
    # Validate that all required columns exist
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    X = df[feature_columns]
    y = df['duration']
    
    # Data quality checks
    logging.info("🔍 Performing data quality checks...")
    logging.info(f"   - Features shape: {X.shape}")
    logging.info(f"   - Target shape: {y.shape}")
    logging.info(f"   - Missing values in features: {X.isnull().sum().sum()}")
    logging.info(f"   - Missing values in target: {y.isnull().sum()}")
    
    # Handle any missing values
    if X.isnull().sum().sum() > 0:
        logging.warning("⚠️  Found missing values in features, filling with median")
        X = X.fillna(X.median())
    
    if y.isnull().sum() > 0:
        logging.warning("⚠️  Found missing values in target, dropping rows")
        valid_indices = ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
    
    # Split the data
    logging.info("✂️  Splitting data into train/test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    logging.info(f"📊 Data split complete:")
    logging.info(f"   - Training set: {len(X_train):,} samples")
    logging.info(f"   - Test set: {len(X_test):,} samples")
    logging.info(f"   - Train/test ratio: {len(X_train)/len(X_test):.1f}:1")
    
    # Train the model
    logging.info("🤖 Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    
    logging.info("✅ Model training completed")
    
    # Make predictions and evaluate
    logging.info("📈 Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Feature importance
    feature_importance = dict(zip(feature_columns, model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'source_records': extraction_results['rows_processed'],
        'feature_importance': feature_importance,
        'model_params': model.get_params(),
        'timestamp': datetime.now().isoformat()
    }
    
    logging.info(f"📊 Model Performance Metrics:")
    logging.info(f"   - RMSE: {rmse:.2f} seconds")
    logging.info(f"   - MAE: {mae:.2f} seconds")
    logging.info(f"   - R² Score: {r2:.4f}")
    logging.info(f"   - MSE: {mse:.2f}")
    
    logging.info(f"🔝 Top Feature Importances:")
    for feature, importance in sorted_features[:5]:
        logging.info(f"   - {feature}: {importance:.4f}")
    
    # Save model to GCS (if bucket is available)
    if bucket is not None:
        logging.info("💾 Saving model to GCS...")
        
        model_filename = f"models/bike_duration_model_{context['ds_nodash']}.joblib"
        local_model_path = f"/tmp/{model_filename.split('/')[-1]}"
        
        try:
            # Save model locally first
            joblib.dump(model, local_model_path)
            
            # Upload to GCS
            blob = bucket.blob(model_filename)
            blob.upload_from_filename(local_model_path)
            
            logging.info(f"✅ Model saved to gs://{bucket_name}/{model_filename}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save model: {str(e)}")
    else:
        logging.warning("⚠️  Skipping model save - no accessible GCS bucket")
        
    # Save metrics to GCS (if bucket is available)
    if bucket is not None:
        logging.info("📊 Saving metrics to GCS...")
        
        metrics_filename = f"metrics/metrics_{context['ds_nodash']}.json"
        local_metrics_path = f"/tmp/{metrics_filename.split('/')[-1]}"
        
        try:
            # Save metrics locally first
            with open(local_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Upload to GCS
            blob = bucket.blob(metrics_filename)
            blob.upload_from_filename(local_metrics_path)
            
            logging.info(f"✅ Metrics saved to gs://{bucket_name}/{metrics_filename}")
            
        except Exception as e:
            logging.error(f"❌ Failed to save metrics: {str(e)}")
    else:
        logging.warning("⚠️  Skipping metrics save - no accessible GCS bucket")
    
    # Clean up temporary files
    try:
        import os
        if os.path.exists(local_model_path):
            os.remove(local_model_path)
        if os.path.exists(local_metrics_path):
            os.remove(local_metrics_path)
    except:
        pass
    
    logging.info("🎉 Model training and persistence completed successfully!")
    
    # Return metrics for downstream tasks
    return metrics

train_model_task = PythonOperator(
    task_id='train_ml_model',
    python_callable=train_model,
    dag=dag,
)

# Task 3: Notification and Summary
def log_completion(**context):
    """Log completion and retrieve metrics from previous task"""
    
    logging.info("🎉 ML Pipeline Execution Complete!")
    logging.info("=" * 60)
    
    try:
        # Get results from previous tasks
        extraction_results = context['ti'].xcom_pull(task_ids='extract_bike_data')
        metrics = context['ti'].xcom_pull(task_ids='train_ml_model')
        
        # Pipeline Summary
        logging.info("📊 PIPELINE EXECUTION SUMMARY:")
        logging.info("=" * 60)
        logging.info(f"🕐 Execution Date: {context['ds']}")
        logging.info(f"🕐 Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logging.info("")
        
        # Data Processing Summary
        logging.info("📈 DATA PROCESSING:")
        logging.info(f"   • Source Records Extracted: {extraction_results['rows_processed']:,}")
        logging.info(f"   • Training Samples: {metrics['training_samples']:,}")
        logging.info(f"   • Test Samples: {metrics['test_samples']:,}")
        logging.info(f"   • Data Source: EU Public Dataset")
        logging.info(f"   • Data Destination: {extraction_results['table_id']}")
        logging.info("")
        
        # Model Performance Summary  
        logging.info("🤖 MODEL PERFORMANCE:")
        logging.info(f"   • Root Mean Square Error: {metrics['rmse']:.2f} seconds")
        logging.info(f"   • Mean Absolute Error: {metrics['mae']:.2f} seconds")
        logging.info(f"   • R² Score: {metrics['r2_score']:.4f}")
        logging.info(f"   • Model Type: Random Forest Regressor")
        logging.info("")
        
        # Feature Importance
        logging.info("🔝 TOP FEATURE IMPORTANCES:")
        sorted_features = sorted(metrics['feature_importance'].items(), 
                               key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            logging.info(f"   {i}. {feature}: {importance:.4f}")
        logging.info("")
        
        # Artifacts Summary
        logging.info("🗂️  SAVED ARTIFACTS:")
        logging.info(f"   • Model: Check 'models/' folder in GCS")
        logging.info(f"   • Metrics: Check 'metrics/' folder in GCS")
        logging.info(f"   • Training Data: {extraction_results['table_id']}")
        logging.info("")
        
        # Performance Interpretation
        r2_score = metrics['r2_score']
        if r2_score >= 0.8:
            performance = "Excellent"
        elif r2_score >= 0.6:
            performance = "Good"
        elif r2_score >= 0.4:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        logging.info("🎯 MODEL ASSESSMENT:")
        logging.info(f"   • Overall Performance: {performance}")
        logging.info(f"   • Explained Variance: {r2_score*100:.1f}%")
        
        avg_duration = (extraction_results['duration_stats']['mean'] / 60)  # Convert to minutes
        rmse_minutes = metrics['rmse'] / 60
        logging.info(f"   • Average Trip Duration: {avg_duration:.1f} minutes")
        logging.info(f"   • Prediction Error: ±{rmse_minutes:.1f} minutes")
        logging.info("")
        
        # Next Steps
        logging.info("🚀 RECOMMENDED NEXT STEPS:")
        if r2_score < 0.6:
            logging.info("   • Consider feature engineering improvements")
            logging.info("   • Experiment with different model parameters")
            logging.info("   • Increase dataset size for better training")
        else:
            logging.info("   • Model ready for production deployment")
            logging.info("   • Consider A/B testing with live data")
            logging.info("   • Monitor model performance over time")
        
        logging.info("")
        logging.info("=" * 60)
        logging.info("✅ Check GCP Console > Cloud Logging for detailed execution logs")
        logging.info("✅ Check GCP Console > Cloud Storage for saved model artifacts")
        logging.info("✅ Check GCP Console > BigQuery for processed training data")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"❌ Error in completion logging: {str(e)}")
        logging.info("🎉 ML Pipeline completed, but summary generation failed")
        logging.info("✅ Check individual task logs for detailed results")

notification_task = PythonOperator(
    task_id='log_completion',
    python_callable=log_completion,
    dag=dag,
)

# Set task dependencies
extract_data_task >> train_model_task >> notification_task