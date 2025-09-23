# backend/app/utils/helpers.py
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import hashlib
import uuid

logger = logging.getLogger(__name__)

class DataValidationHelper:
    """Helper functions for data validation and transformation"""
    
    @staticmethod
    def validate_trainset_id(trainset_id: str) -> bool:
        """Validate trainset ID format"""
        if not trainset_id:
            return False
        
        # Expected format: T-001, T-002, etc.
        if not trainset_id.startswith('T-'):
            return False
        
        try:
            number_part = trainset_id[2:]
            int(number_part)
            return len(number_part) == 3
        except ValueError:
            return False
    
    @staticmethod
    def validate_fitness_certificate(cert_data: Dict[str, Any]) -> bool:
        """Validate fitness certificate data"""
        required_fields = ['status', 'expiry_date', 'issued_by', 'certificate_id']
        
        for field in required_fields:
            if field not in cert_data:
                return False
        
        # Validate status
        valid_statuses = ['VALID', 'EXPIRED', 'EXPIRING_SOON']
        if cert_data['status'] not in valid_statuses:
            return False
        
        return True
    
    @staticmethod
    def validate_mileage_data(mileage: float, max_mileage: float) -> bool:
        """Validate mileage data"""
        if mileage < 0 or max_mileage <= 0:
            return False
        
        if mileage > max_mileage * 1.1:  # Allow 10% tolerance
            return False
        
        return True

class DataTransformationHelper:
    """Helper functions for data transformation"""
    
    @staticmethod
    def transform_trainset_for_api(trainset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trainset data for API response"""
        # Remove internal fields
        internal_fields = ['_id', 'internal_notes', 'debug_info']
        
        for field in internal_fields:
            trainset_data.pop(field, None)
        
        # Add computed fields
        trainset_data['fitness_score'] = DataTransformationHelper._calculate_fitness_score(
            trainset_data.get('fitness_certificates', {})
        )
        
        trainset_data['maintenance_priority'] = DataTransformationHelper._calculate_maintenance_priority(
            trainset_data.get('job_cards', {})
        )
        
        return trainset_data
    
    @staticmethod
    def _calculate_fitness_score(fitness_certs: Dict[str, Any]) -> float:
        """Calculate overall fitness score"""
        if not fitness_certs:
            return 0.0
        
        valid_count = sum(1 for cert in fitness_certs.values() if cert.get('status') == 'VALID')
        total_count = len(fitness_certs)
        
        return valid_count / total_count if total_count > 0 else 0.0
    
    @staticmethod
    def _calculate_maintenance_priority(job_cards: Dict[str, Any]) -> str:
        """Calculate maintenance priority based on job cards"""
        critical_cards = job_cards.get('critical_cards', 0)
        open_cards = job_cards.get('open_cards', 0)
        
        if critical_cards > 0:
            return 'CRITICAL'
        elif open_cards > 5:
            return 'HIGH'
        elif open_cards > 2:
            return 'MEDIUM'
        else:
            return 'LOW'

class CacheHelper:
    """Helper functions for caching operations"""
    
    @staticmethod
    def generate_cache_key(prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        key_parts = [prefix]
        
        for key, value in sorted(kwargs.items()):
            if value is not None:
                key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    @staticmethod
    def serialize_for_cache(data: Any) -> str:
        """Serialize data for caching"""
        if isinstance(data, dict):
            return json.dumps(data, default=str)
        return str(data)
    
    @staticmethod
    def deserialize_from_cache(cached_data: str) -> Any:
        """Deserialize data from cache"""
        try:
            return json.loads(cached_data)
        except (json.JSONDecodeError, TypeError):
            return cached_data

class SecurityHelper:
    """Helper functions for security operations"""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate API key"""
        return f"kmrl-api-{uuid.uuid4().hex[:16]}"
    
    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Hash sensitive data"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key:
            return False
        
        # Expected format: kmrl-api-xxxxxxxxxxxxxxxx
        parts = api_key.split('-')
        return len(parts) == 3 and parts[0] == 'kmrl' and parts[1] == 'api' and len(parts[2]) == 16

class TimeHelper:
    """Helper functions for time operations"""
    
    @staticmethod
    def is_expired(expiry_date: str) -> bool:
        """Check if date is expired"""
        try:
            expiry = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
            return expiry < datetime.now()
        except (ValueError, TypeError):
            return True
    
    @staticmethod
    def is_expiring_soon(expiry_date: str, days_threshold: int = 30) -> bool:
        """Check if date is expiring soon"""
        try:
            expiry = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
            threshold = datetime.now() + timedelta(days=days_threshold)
            return datetime.now() < expiry < threshold
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def format_timestamp(timestamp: Union[str, datetime]) -> str:
        """Format timestamp to ISO format"""
        if isinstance(timestamp, str):
            return timestamp
        elif isinstance(timestamp, datetime):
            return timestamp.isoformat()
        else:
            return datetime.now().isoformat()

class ResponseHelper:
    """Helper functions for API responses"""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Create success response"""
        return {
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def error_response(error: str, status_code: int = 400) -> Dict[str, Any]:
        """Create error response"""
        return {
            "status": "error",
            "error": error,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def paginated_response(data: List[Any], page: int, limit: int, total: int) -> Dict[str, Any]:
        """Create paginated response"""
        return {
            "data": data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": (total + limit - 1) // limit
            },
            "timestamp": datetime.now().isoformat()
        }
