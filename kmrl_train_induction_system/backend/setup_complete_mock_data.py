#!/usr/bin/env python3
"""
Complete Mock Data Setup for KMRL Train Induction System
This script creates all necessary mock data for the frontend to work properly.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.cloud_database import cloud_db_manager
from app.models.assignment import Assignment, AssignmentStatus, Decision
from app.models.trainset import Trainset
from app.models.audit import AuditLog, AuditAction
from app.models.notification import Notification, NotificationType

class CompleteMockDataGenerator:
    def __init__(self):
        self.trainset_ids = [
            "TS-001", "TS-002", "TS-003", "TS-004", "TS-005",
            "TS-006", "TS-007", "TS-008", "TS-009", "TS-010",
            "TS-011", "TS-012", "TS-013", "TS-014", "TS-015"
        ]
        
    async def generate_trainsets(self) -> List[Dict[str, Any]]:
        """Generate mock trainset data"""
        trainsets = []
        for i, trainset_id in enumerate(self.trainset_ids):
            trainset = {
                "trainset_id": trainset_id,
                "status": random.choice(["ACTIVE", "MAINTENANCE", "STANDBY"]),
                "sensor_health_score": round(random.uniform(0.7, 1.0), 2),
                "predicted_failure_risk": round(random.uniform(0.1, 0.4), 2),
                "branding": {
                    "priority": random.choice(["HIGH", "MEDIUM", "LOW"]),
                    "contract_valid": random.choice([True, False])
                },
                "certificates": {
                    "safety": {
                        "valid": random.choice([True, False]),
                        "expiry_date": (datetime.now() + timedelta(days=random.randint(-30, 90))).isoformat()
                    },
                    "maintenance": {
                        "valid": random.choice([True, False]),
                        "expiry_date": (datetime.now() + timedelta(days=random.randint(-30, 90))).isoformat()
                    }
                },
                "cleaning_schedule": {
                    "last_cleaning": (datetime.now() - timedelta(days=random.randint(1, 7))).isoformat(),
                    "next_cleaning": (datetime.now() + timedelta(days=random.randint(1, 3))).isoformat()
                },
                "mileage": {
                    "total_km": random.randint(50000, 200000),
                    "last_30_days": random.randint(1000, 5000)
                },
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            trainsets.append(trainset)
        return trainsets

    async def generate_assignments(self) -> List[Dict[str, Any]]:
        """Generate mock assignment data"""
        assignments = []
        decisions = ["INDUCT", "STANDBY", "MAINTENANCE"]
        statuses = ["PENDING", "APPROVED", "OVERRIDDEN"]
        
        for i, trainset_id in enumerate(self.trainset_ids):
            decision = random.choice(decisions)
            status = random.choice(statuses)
            confidence = round(random.uniform(0.6, 0.95), 2)
            
            # Generate violations for some assignments
            violations = []
            if random.random() < 0.3:  # 30% chance of violations
                violation_types = [
                    "Safety certificate expiring soon",
                    "Maintenance overdue",
                    "Cleaning schedule conflict",
                    "Branding contract expired"
                ]
                violations = random.sample(violation_types, random.randint(1, 2))
            
            assignment = {
                "id": f"ASS-{i+1:03d}",
                "trainset_id": trainset_id,
                "status": status,
                "priority": random.randint(1, 5),
                "decision": {
                    "decision": decision,
                    "confidence_score": confidence,
                    "reasoning": f"AI decision based on trainset {trainset_id} analysis",
                    "violations": violations
                },
                "created_at": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat(),
                "updated_at": datetime.now().isoformat(),
                "assigned_to": f"Operator-{random.randint(1, 5)}",
                "scheduled_date": (datetime.now() + timedelta(days=random.randint(1, 3))).isoformat()
            }
            assignments.append(assignment)
        return assignments

    async def generate_ranked_induction_list(self) -> List[Dict[str, Any]]:
        """Generate mock ranked induction list"""
        decisions = []
        for i, trainset_id in enumerate(self.trainset_ids):
            decision = random.choice(["INDUCT", "STANDBY", "MAINTENANCE"])
            score = round(random.uniform(0.6, 0.95), 3)
            confidence = round(random.uniform(0.7, 0.95), 2)
            
            # Generate top reasons
            reasons = [
                "All department certificates valid",
                "Low predicted failure probability",
                "Available cleaning slot before dawn",
                "High sensor health score",
                "Recent maintenance completed",
                "Branding contract active"
            ]
            top_reasons = random.sample(reasons, random.randint(2, 4))
            
            # Generate risks
            risks = [
                "Safety certificate expiring soon",
                "High mileage in last 30 days",
                "Cleaning schedule conflict",
                "Branding contract expiring"
            ]
            top_risks = random.sample(risks, random.randint(0, 2))
            
            # Generate SHAP values
            shap_values = [
                {"name": "Sensor Health Score", "value": round(random.uniform(0.7, 1.0), 2), "impact": "positive"},
                {"name": "Predicted Failure Risk", "value": round(random.uniform(0.1, 0.4), 2), "impact": "positive"},
                {"name": "Branding Priority", "value": round(random.uniform(0.3, 1.0), 2), "impact": "positive"},
                {"name": "Maintenance Status", "value": round(random.uniform(0.6, 1.0), 2), "impact": "positive"}
            ]
            
            decision_data = {
                "trainset_id": trainset_id,
                "decision": decision,
                "confidence_score": confidence,
                "score": score,
                "top_reasons": top_reasons,
                "top_risks": top_risks,
                "violations": [],
                "shap_values": shap_values,
                "reasons": top_reasons
            }
            decisions.append(decision_data)
        
        # Sort by score (highest first)
        decisions.sort(key=lambda x: x["score"], reverse=True)
        return decisions

    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate mock dashboard data"""
        return {
            "fleet_status": {
                "active": 12,
                "maintenance": 2,
                "standby": 1
            },
            "system_health": {
                "api_response_time_ms": random.randint(50, 200),
                "database_connections": random.randint(8, 12),
                "queue_size": random.randint(0, 5)
            },
            "operational_metrics": {
                "fleet_availability": round(random.uniform(0.85, 0.95), 2),
                "energy_efficiency": round(random.uniform(0.80, 0.90), 2),
                "punctuality_rate": round(random.uniform(0.90, 0.98), 2)
            }
        }

    async def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate mock alerts data"""
        alerts = []
        alert_types = ["CRITICAL", "WARNING", "INFO"]
        messages = [
            "Safety certificate expiring for TS-003",
            "Maintenance overdue for TS-007",
            "Cleaning schedule conflict detected",
            "Branding contract expired for TS-012",
            "High failure risk predicted for TS-005"
        ]
        
        for i in range(5):
            alert = {
                "id": f"ALERT-{i+1:03d}",
                "type": random.choice(alert_types),
                "message": random.choice(messages),
                "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
                "category": random.choice(["SAFETY", "MAINTENANCE", "OPERATIONS"]),
                "trainset_id": random.choice(self.trainset_ids),
                "resolved": random.choice([True, False])
            }
            alerts.append(alert)
        return alerts

    async def generate_reports_data(self) -> Dict[str, Any]:
        """Generate mock reports data"""
        return {
            "daily_briefing": {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_assignments": 15,
                "pending_approvals": 3,
                "completed_today": 12,
                "critical_alerts": 2
            },
            "fleet_status": {
                "total_trainsets": 15,
                "active": 12,
                "maintenance": 2,
                "standby": 1
            },
            "performance_analysis": {
                "efficiency_score": 0.87,
                "reliability_score": 0.92,
                "safety_score": 0.95
            }
        }

    async def setup_all_data(self):
        """Setup all mock data in the database"""
        try:
            print("🚀 Setting up complete mock data for KMRL Operations Dashboard...")
            
            # Connect to database
            await cloud_db_manager.connect_mongodb()
            print("✅ Connected to MongoDB")
            
            # Generate and insert trainsets
            trainsets = await self.generate_trainsets()
            trainsets_collection = await cloud_db_manager.get_collection("trainsets")
            await trainsets_collection.delete_many({})
            await trainsets_collection.insert_many(trainsets)
            print(f"✅ Inserted {len(trainsets)} trainsets")
            
            # Generate and insert assignments
            assignments = await self.generate_assignments()
            assignments_collection = await cloud_db_manager.get_collection("assignments")
            await assignments_collection.delete_many({})
            await assignments_collection.insert_many(assignments)
            print(f"✅ Inserted {len(assignments)} assignments")
            
            # Generate and insert ranked induction list
            ranked_list = await self.generate_ranked_induction_list()
            latest_collection = await cloud_db_manager.get_collection("latest_induction")
            await latest_collection.delete_many({})
            await latest_collection.insert_one({
                "decisions": ranked_list,
                "created_at": datetime.now().isoformat(),
                "optimization_run_id": f"OPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            })
            print(f"✅ Inserted ranked induction list with {len(ranked_list)} decisions")
            
            # Generate and insert dashboard data
            dashboard_data = await self.generate_dashboard_data()
            dashboard_collection = await cloud_db_manager.get_collection("dashboard")
            await dashboard_collection.delete_many({})
            await dashboard_collection.insert_one(dashboard_data)
            print("✅ Inserted dashboard data")
            
            # Generate and insert alerts
            alerts = await self.generate_alerts()
            alerts_collection = await cloud_db_manager.get_collection("alerts")
            await alerts_collection.delete_many({})
            await alerts_collection.insert_many(alerts)
            print(f"✅ Inserted {len(alerts)} alerts")
            
            # Generate and insert reports data
            reports_data = await self.generate_reports_data()
            reports_collection = await cloud_db_manager.get_collection("reports")
            await reports_collection.delete_many({})
            await reports_collection.insert_one(reports_data)
            print("✅ Inserted reports data")
            
            print("\n🎉 Complete mock data setup finished!")
            print("All endpoints should now work properly.")
            
        except Exception as e:
            print(f"❌ Error setting up mock data: {e}")
            raise

async def main():
    generator = CompleteMockDataGenerator()
    await generator.setup_all_data()

if __name__ == "__main__":
    asyncio.run(main())
