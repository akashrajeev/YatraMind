"""Application service for dashboard read models."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.repositories.mongo_dashboard import MongoDashboardRepository


class DashboardService:
    """Build dashboard views without exposing database details to the API."""

    def __init__(self, repository: MongoDashboardRepository | None = None) -> None:
        self.repository = repository or MongoDashboardRepository()

    async def overview(self) -> dict[str, Any]:
        trainsets = [dict(item) for item in await self.repository.list_trainsets()]
        latest = await self.repository.get_latest_induction()

        active = sum(item.get("status") == "ACTIVE" for item in trainsets)
        maintenance = sum(item.get("status") == "MAINTENANCE" for item in trainsets)
        standby = sum(item.get("status") == "STANDBY" for item in trainsets)
        decisions = (latest or {}).get("decisions") or []
        if decisions:
            active = sum(item.get("decision") == "INDUCT" for item in decisions if isinstance(item, Mapping))
            maintenance = sum(item.get("decision") == "MAINTENANCE" for item in decisions if isinstance(item, Mapping))
            standby = sum(item.get("decision") == "STANDBY" for item in decisions if isinstance(item, Mapping))

        valid = expired = expiring = open_cards = critical_cards = 0
        depots: dict[str, int] = {}
        for trainset in trainsets:
            for cert in (trainset.get("fitness_certificates") or {}).values():
                status = str(cert.get("status", "")).upper() if isinstance(cert, Mapping) else ""
                if status == "VALID":
                    valid += 1
                elif status == "EXPIRED":
                    expired += 1
                else:
                    expiring += 1
            cards = trainset.get("job_cards") or {}
            open_cards += int(cards.get("open_cards", 0) or 0)
            critical_cards += int(cards.get("critical_cards", 0) or 0)
            location = trainset.get("current_location") or {}
            depot = location.get("depot") if isinstance(location, Mapping) else None
            if depot:
                depots[str(depot)] = depots.get(str(depot), 0) + 1

        assigned_ids = await self.repository.list_assigned_trainset_ids() if latest else set()
        virtual_pending = sum(
            1
            for decision in decisions
            if isinstance(decision, Mapping)
            and decision.get("trainset_id")
            and decision["trainset_id"] not in assigned_ids
        )

        return {
            "total_trainsets": len(trainsets),
            "fleet_status": {"active": active, "maintenance": maintenance, "standby": standby},
            "fitness_certificates": {"valid": valid, "expired": expired, "expiring_soon": expiring},
            "job_cards": {"total_open": open_cards, "critical": critical_cards},
            "depot_distribution": depots,
            "sensor_health": {"average_health_score": 0.87, "sensors_online": 98, "sensors_critical": 2},
            "pending_assignments": await self.repository.count_pending_assignments() + virtual_pending,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    async def alerts(self) -> dict[str, Any]:
        candidates = await self.repository.list_alert_candidates()
        alerts: list[dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for item in candidates:
            trainset_id = item.get("trainset_id")
            for cert_type, cert in (item.get("fitness_certificates") or {}).items():
                status = str(cert.get("status", "")).upper() if isinstance(cert, Mapping) else ""
                if status in {"EXPIRED", "EXPIRING_SOON"}:
                    alerts.append({
                        "type": "CRITICAL" if status == "EXPIRED" else "WARNING",
                        "category": "CERTIFICATE",
                        "trainset_id": trainset_id,
                        "message": f"{cert_type} certificate has expired" if status == "EXPIRED" else f"{cert_type} certificate expires soon",
                        "timestamp": timestamp,
                    })
            critical = int((item.get("job_cards") or {}).get("critical_cards", 0) or 0)
            if critical:
                alerts.append({
                    "type": "HIGH",
                    "category": "MAINTENANCE",
                    "trainset_id": trainset_id,
                    "message": f"{critical} critical job cards pending",
                    "timestamp": timestamp,
                })
            mileage = float(item.get("current_mileage", 0) or 0)
            maximum = float(item.get("max_mileage_before_maintenance", 100000) or 100000)
            if mileage >= maximum * 0.95:
                alerts.append({
                    "type": "WARNING",
                    "category": "MILEAGE",
                    "trainset_id": trainset_id,
                    "message": f"Approaching mileage limit: {mileage} km",
                    "timestamp": timestamp,
                })

        severity = {"CRITICAL": 0, "HIGH": 1, "WARNING": 2}
        alerts.sort(key=lambda item: severity.get(item["type"], 3))
        return {
            "total_alerts": len(alerts),
            "critical_count": sum(item["type"] == "CRITICAL" for item in alerts),
            "high_count": sum(item["type"] == "HIGH" for item in alerts),
            "warning_count": sum(item["type"] == "WARNING" for item in alerts),
            "alerts": alerts[:20],
        }

    async def performance(self) -> dict[str, Any]:
        history = [dict(item) for item in await self.repository.get_recent_optimization_history(7)]
        average_confidence = (
            sum(float(item.get("average_confidence", 0) or 0) for item in history) / len(history)
            if history else 0.0
        )
        return {
            "optimization_performance": {
                "total_runs": len(history),
                "average_confidence_score": round(average_confidence, 2),
                "recent_history": history,
            },
            "operational_metrics": {
                "punctuality_rate": 99.7,
                "fleet_availability": 96.2,
                "energy_efficiency": 87.5,
                "maintenance_cost_reduction": 12.3,
            },
            "sensor_analytics": {
                "temperature_trends": [],
                "vibration_analysis": [],
                "predictive_maintenance_alerts": 3,
            },
            "system_health": {
                "api_response_time_ms": 45,
                "optimization_time_seconds": 8.2,
                "database_performance": "GOOD",
                "mqtt_connectivity": "ONLINE",
            },
        }
