# backend/app/services/data_cleaning.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataCleaningService:
    """Data cleaning and validation using Pandas + NumPy"""
    
    def __init__(self):
        self.cleaning_rules = {
            "remove_duplicates": True,
            "handle_missing_values": True,
            "validate_data_types": True,
            "outlier_detection": True
        }
    
    def clean_trainset_data(self, trainsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate trainset data using Pandas"""
        try:
            if not trainsets:
                return trainsets
            
            # Convert to DataFrame for cleaning
            df = pd.DataFrame(trainsets)
            
            # Remove duplicates based on trainset_id
            if self.cleaning_rules["remove_duplicates"]:
                df = df.drop_duplicates(subset=['trainset_id'], keep='first')
                logger.info(f"Removed {len(trainsets) - len(df)} duplicate trainsets")
            
            # Handle missing values
            if self.cleaning_rules["handle_missing_values"]:
                df = self._handle_missing_values(df)
            
            # Validate data types
            if self.cleaning_rules["validate_data_types"]:
                df = self._validate_data_types(df)
            
            # Detect outliers
            if self.cleaning_rules["outlier_detection"]:
                df = self._detect_outliers(df)
            
            # Convert back to list of dictionaries
            cleaned_trainsets = df.to_dict('records')
            
            logger.info(f"Data cleaning completed: {len(cleaned_trainsets)} trainsets processed")
            return cleaned_trainsets
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return trainsets  # Return original data if cleaning fails
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill missing status with 'STANDBY'
        df['status'] = df['status'].fillna('STANDBY')
        
        # Fill missing mileage with 0
        df['current_mileage'] = df['current_mileage'].fillna(0)
        
        # Fill missing max_mileage with default
        df['max_mileage_before_maintenance'] = df['max_mileage_before_maintenance'].fillna(50000)
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types"""
        # Ensure mileage is numeric
        df['current_mileage'] = pd.to_numeric(df['current_mileage'], errors='coerce')
        df['max_mileage_before_maintenance'] = pd.to_numeric(df['max_mileage_before_maintenance'], errors='coerce')
        
        # Ensure status is string
        df['status'] = df['status'].astype(str)
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numerical data"""
        # Detect outliers in mileage using IQR method
        if 'current_mileage' in df.columns:
            Q1 = df['current_mileage'].quantile(0.25)
            Q3 = df['current_mileage'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df['current_mileage'] = np.where(
                df['current_mileage'] < lower_bound, lower_bound,
                np.where(df['current_mileage'] > upper_bound, upper_bound, df['current_mileage'])
            )
        
        return df
    
    def clean_sensor_data(self, sensor_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean IoT sensor data"""
        try:
            if not sensor_data:
                return sensor_data
            
            df = pd.DataFrame(sensor_data)
            
            # Remove invalid sensor readings
            df = df[df['health_score'].between(0, 1)]
            df = df[df['temperature'].between(-50, 100)]  # Reasonable temperature range
            
            # Remove duplicates based on timestamp and sensor_id
            df = df.drop_duplicates(subset=['trainset_id', 'sensor_type', 'timestamp'])
            
            # Calculate sensor health score
            df['sensor_health_score'] = df.groupby('trainset_id')['health_score'].transform('mean')
            
            cleaned_data = df.to_dict('records')
            logger.info(f"Sensor data cleaning completed: {len(cleaned_data)} readings processed")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Sensor data cleaning failed: {e}")
            return sensor_data
