# backend/app/services/data_cleaning.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from app.utils.cloud_database import cloud_db_manager
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

    # --------------------------- Feature Engineering (ETL) --------------------------- #
    def build_fitness_windows(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Manage missing fitness fields, convert to UTC, compute valid_at_dawn.

        Expected fields per record: trainset_id, certificate, valid_from, valid_to, status
        """
        if not records:
            return []
        df = pd.DataFrame(records).copy()
        # Standardize columns
        for c in ["trainset_id", "certificate", "valid_from", "valid_to", "status"]:
            if c not in df.columns:
                df[c] = np.nan

        # Mark invalid where critical fields missing
        df["status"] = df["status"].fillna("INVALID")
        missing_mask = df[["trainset_id", "certificate", "valid_to"]].isna().any(axis=1)
        df.loc[missing_mask, "status"] = "INVALID"

        # Convert to UTC timestamps
        def _to_utc(series: pd.Series) -> pd.Series:
            ts = pd.to_datetime(series, errors="coerce")
            # Localize naive timestamps to Asia/Kolkata and convert to UTC
            try:
                ts = ts.dt.tz_localize("Asia/Kolkata")
            except TypeError:
                # Already tz-aware
                pass
            try:
                ts = ts.dt.tz_convert("UTC")
            except Exception:
                # If still naive, localize to UTC directly
                try:
                    ts = ts.dt.tz_localize("UTC")
                except Exception:
                    pass
            return ts

        df["valid_from_utc"] = _to_utc(df["valid_from"]).astype("datetime64[ns, UTC]")
        df["valid_to_utc"] = _to_utc(df["valid_to"]).astype("datetime64[ns, UTC]")

        # Compute next dawn in Asia/Kolkata, then convert to UTC
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        dawn_ist = (now_ist if now_ist.hour < 5 else now_ist + timedelta(days=1)).replace(hour=5, minute=0, second=0, microsecond=0)
        dawn_utc = dawn_ist.astimezone(ZoneInfo("UTC"))
        df["valid_at_dawn"] = (df["valid_to_utc"].dt.tz_convert("UTC") >= dawn_utc).fillna(False)

        out_cols = [
            "trainset_id",
            "certificate",
            "status",
            "valid_from_utc",
            "valid_to_utc",
            "valid_at_dawn",
        ]
        return df[out_cols].to_dict(orient="records")

    def normalize_job_cards(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize job-card statuses and extract criticality."""
        if not records:
            return []
        df = pd.DataFrame(records).copy()
        df.columns = df.columns.str.lower()
        # Status map
        status_map = {
            "open": "OPEN",
            "inprogress": "OPEN",
            "wip": "OPEN",
            "closed": "CLOSED",
            "comp": "CLOSED",
        }
        df["status"] = df["status"].astype(str).str.lower().map(status_map).fillna("OPEN")
        # Criticality
        def _critical(row) -> bool:
            pr = str(row.get("priority", "")).upper()
            return pr in {"HIGH", "CRITICAL"} or row.get("critical", False) or (row.get("wopriority", 0) in (1, 2))
        df["critical"] = df.apply(_critical, axis=1)
        # Severity bucket
        df["severity"] = np.where(df["critical"], "CRITICAL", np.where(df["status"] == "OPEN", "MAJOR", "MINOR"))
        cols = ["job_card_id", "trainset_id", "status", "severity", "critical", "estimated_cost", "estimated_duration_hours", "created_date"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols].to_dict(orient="records")

    def mileage_rolling(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize mileage logs to 30/90 day rolling sums per rake."""
        if not logs:
            return []
        df = pd.DataFrame(logs).copy()
        df.columns = df.columns.str.lower()
        for c in ["trainset_id", "date", "kilometers"]:
            if c not in df.columns:
                df[c] = np.nan
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["trainset_id", "date"])  # ensure order
        df["km_30d"] = (
            df.set_index("date")
            .groupby("trainset_id")["kilometers"]
            .rolling("30D", min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .values
        )
        df["km_90d"] = (
            df.set_index("date")
            .groupby("trainset_id")["kilometers"]
            .rolling("90D", min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .values
        )
        out = df[["trainset_id", "date", "kilometers", "km_30d", "km_90d"]]
        return out.to_dict(orient="records")

    def cleaning_slot_capacity(self, schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map cleaning schedules into rake_id -> availability before dawn."""
        if not schedule:
            return []
        df = pd.DataFrame(schedule).copy()
        df.columns = df.columns.str.lower()
        for c in ["trainset_id", "date", "slot", "bay"]:
            if c not in df.columns:
                df[c] = np.nan
        # Interpret slot format like "22:00-02:00"
        def _slot_ends_before_dawn(row) -> bool:
            try:
                date = pd.to_datetime(row["date"]).date()
                start_s, end_s = str(row["slot"]).split("-")
                # assume slots can cross midnight; create start at date and end at date or next date
                start = pd.Timestamp(f"{date} {start_s}", tz=ZoneInfo("Asia/Kolkata"))
                end = pd.Timestamp(f"{date} {end_s}", tz=ZoneInfo("Asia/Kolkata"))
                if end < start:
                    end = end + pd.Timedelta(days=1)
                dawn = pd.Timestamp.combine(end.date(), pd.Timestamp("05:00").time()).replace(tzinfo=ZoneInfo("Asia/Kolkata"))
                return end <= dawn
            except Exception:
                return False
        df["available_before_dawn"] = df.apply(_slot_ends_before_dawn, axis=1)
        # Aggregate per trainset (any availability)
        agg = (
            df.groupby(["trainset_id", "bay"], dropna=False)["available_before_dawn"]
            .any()
            .reset_index()
        )
        agg.rename(columns={"bay": "target_bay"}, inplace=True)
        return agg.to_dict(orient="records")

    def derive_stabling_and_shunt_cost(
        self,
        rakes: List[Dict[str, Any]],
        cleaning_capacity: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Derive stabling position and a simple shunt cost to a target bay.

        Heuristic: if current bay and target bay are numeric-like, cost = abs(diff).
        """
        df_r = pd.DataFrame(rakes)
        df_c = pd.DataFrame(cleaning_capacity)
        if df_r.empty:
            return []
        for c in ["trainset_id", "current_location"]:
            if c not in df_r.columns:
                df_r[c] = np.nan
        # Extract current bay number if present
        def _bay_num(val) -> float:
            try:
                if isinstance(val, dict):
                    bay = val.get("bay", "")
                else:
                    bay = val
                return float("".join([ch for ch in str(bay) if ch.isdigit()]))
            except Exception:
                return np.nan
        df_r["current_bay_num"] = df_r["current_location"].apply(_bay_num)
        if not df_c.empty:
            # Merge target bay per trainset if available
            merged = df_r.merge(df_c[["trainset_id", "target_bay"]], on="trainset_id", how="left")
        else:
            merged = df_r.copy()
            merged["target_bay"] = np.nan
        merged["target_bay_num"] = merged["target_bay"].apply(_bay_num)
        merged["shunt_cost"] = (merged["current_bay_num"] - merged["target_bay_num"]).abs()
        merged.rename(columns={"current_location": "stabling_position"}, inplace=True)
        return merged[["trainset_id", "stabling_position", "target_bay", "shunt_cost"]].to_dict(orient="records")

    async def persist_clean_features(
        self,
        *,
        rakes: List[Dict[str, Any]] | None = None,
        fitness_windows: List[Dict[str, Any]] | None = None,
        job_cards: List[Dict[str, Any]] | None = None,
        branding_requirements: List[Dict[str, Any]] | None = None,
        cleaning_capacity: List[Dict[str, Any]] | None = None,
        telemetry_features: List[Dict[str, Any]] | None = None,
        depot_geometry: List[Dict[str, Any]] | Dict[str, Any] | None = None,
    ) -> None:
        """Persist cleaned features into MongoDB collections with standard names."""
        async def _upsert_all(col_name: str, docs: List[Dict[str, Any]]):
            if not docs:
                return
            col = await cloud_db_manager.get_collection(col_name)
            if col_name in {"rakes", "fitness_windows", "job_cards", "branding_requirements", "cleaning_capacity", "telemetry_features"}:
                await col.delete_many({})
            await col.insert_many(docs)

        await _upsert_all("rakes", rakes or [])
        await _upsert_all("fitness_windows", fitness_windows or [])
        await _upsert_all("job_cards", job_cards or [])
        await _upsert_all("branding_requirements", branding_requirements or [])
        await _upsert_all("cleaning_capacity", cleaning_capacity or [])
        await _upsert_all("telemetry_features", telemetry_features or [])
        if depot_geometry is not None:
            col = await cloud_db_manager.get_collection("depot_geometry")
            await col.delete_many({})
            if isinstance(depot_geometry, list):
                if depot_geometry:
                    await col.insert_many(depot_geometry)
            else:
                await col.insert_one(depot_geometry)
