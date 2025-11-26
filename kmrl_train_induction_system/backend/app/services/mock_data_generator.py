import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid
import asyncio
from app.utils.cloud_database import cloud_db_manager
from app.config import settings
from pathlib import Path

class KMRLMockDataGenerator:
    """Generate realistic KMRL mock data following technical architecture"""
    
    def __init__(self):
        self.depots = {
            "Aluva": {
                "coordinates": [76.3527, 10.1081],
                "total_bays": 8,
                "maintenance_bays": 3,
                "cleaning_bays": 2
            },
            "Petta": {
                "coordinates": [76.2673, 9.9312], 
                "total_bays": 6,
                "maintenance_bays": 2,
                "cleaning_bays": 1
            }
        }
        
        self.trainset_ids = [f"T-{str(i).zfill(3)}" for i in range(1, 26)]  # T-001 to T-025
        
        self.sensor_types = [
            "bogie_monitoring", "brake_system", "hvac_control", 
            "door_mechanism", "pantograph", "traction_motor",
            "auxiliary_power", "passenger_info_system"
        ]

    async def generate_all_mock_data(self):
        """Generate all mock data following the technical architecture flow"""
        print("Generating KMRL Mock Data following Technical Architecture...")
        
        # Generate trainsets
        trainsets = await self.generate_trainsets_data()
        
        # Generate job cards
        job_cards = await self.generate_maximo_job_cards()
        
        # Generate branding contracts
        branding_records = await self.generate_branding_contracts()
        
        # Generate sensor data
        sensor_data = await self.generate_iot_sensor_data()
        
        # Generate cleaning schedule
        cleaning_schedule = await self.generate_cleaning_schedule()
        
        # Generate historical operations
        historical_data = await self.generate_historical_operations()
        
        # Generate depot layout
        depot_layout = await self.generate_depot_layout_geojson()
        
        data = {
            "depot_layout": depot_layout,
            "trainsets": trainsets,
            "job_cards": job_cards,
            "branding_records": branding_records,
            "sensor_data": sensor_data,
            "cleaning_schedule": cleaning_schedule,
            "historical_operations": historical_data
        }

        # Persist to backend/data/mock for dev server to read if needed
        out_dir = Path(__file__).resolve().parents[3] / "data" / "mock"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trainsets.json").write_text(json.dumps(trainsets, indent=2))
        (out_dir / "maximo_job_cards.json").write_text(json.dumps(job_cards, indent=2))
        (out_dir / "cleaning_schedule.json").write_text(json.dumps(cleaning_schedule, indent=2))
        (out_dir / "historical_operations.json").write_text(json.dumps(historical_data, indent=2))

        return data

    async def generate_trainsets_data(self):
        """Generate trainsets with fitness certificates and realistic health mix.

        Realistic Operational Mix:
        - Group A (Golden Fleet)   : 60%  -> Healthy, pass Safety Gate
        - Group B (Standby)        : 20%  -> Safety OK, minor defects
        - Group C (Critical Fail)  : 20%  -> Fail Safety Gate (expired cert OR critical cards)

        Within Group A:
        - ~30% have REQUIRED branding (Tier 2 test)
        - ~30% have mileage close to limit (Tier 3 test)
        """
        trainsets: List[Dict[str, Any]] = []

        total = len(self.trainset_ids)
        golden_count = int(total * 0.6)
        standby_count = int(total * 0.2)
        critical_count = total - golden_count - standby_count  # ensure full coverage

        def _valid_cert(status_override: str | None = None, min_days: int = 30, max_days: int = 365) -> Dict[str, Any]:
            status = status_override or "VALID"
            return {
                "status": status,
                "expiry_date": (datetime.now() + timedelta(days=random.randint(min_days, max_days))).isoformat(),
                "issued_by": "RDSO",
                "certificate_id": f"GEN_{random.randint(1000, 9999)}",
            }

        for i, trainset_id in enumerate(self.trainset_ids):
            depot = random.choice(list(self.depots.keys()))
            bay_num = random.randint(1, self.depots[depot]["total_bays"])

            # Determine group based on index
            if i < golden_count:
                group = "A"  # Golden Fleet
            elif i < golden_count + standby_count:
                group = "B"  # Standby candidates
            else:
                group = "C"  # Critical failures

            # ---------- Fitness Certificates ----------
            if group in ("A", "B"):
                # All certificates VALID
                fitness_certs = {
                    "rolling_stock": _valid_cert("VALID", 90, 365),
                    "signalling": _valid_cert("VALID", 60, 365),
                    "telecom": _valid_cert("VALID", 60, 365),
                }
            else:
                # Start with valid, then introduce one safety‑critical issue
                fitness_certs = {
                    "rolling_stock": _valid_cert("VALID", 30, 365),
                    "signalling": _valid_cert("VALID", 30, 365),
                    "telecom": _valid_cert("VALID", 30, 365),
                }
                # 50% of critical fleet via expired certificate
                if random.random() < 0.5:
                    cert_to_expire = random.choice(["rolling_stock", "signalling", "telecom"])
                    fitness_certs[cert_to_expire]["status"] = "EXPIRED"
                    fitness_certs[cert_to_expire]["expiry_date"] = (
                        datetime.now() - timedelta(days=random.randint(1, 60))
                    ).isoformat()

            # ---------- Job Cards (open / critical) ----------
            if group == "A":
                # 0 critical, 0–2 minor
                critical_cards = 0
                open_cards = random.randint(0, 2)
            elif group == "B":
                # 0 critical, 3–5 minor
                critical_cards = 0
                open_cards = random.randint(3, 5)
            else:
                # Critical failures: 1+ critical OR already expired cert.
                # Ensure at least some trains have critical job cards as well.
                critical_cards = random.randint(1, 3)
                open_cards = critical_cards + random.randint(0, 5)

            job_cards = {
                "open_cards": open_cards,
                "critical_cards": critical_cards,
            }

            # ---------- Mileage (Tier 3) ----------
            max_mileage = random.choice([50000, 52000, 55000])

            if group == "A":
                # Golden fleet: mix of low, medium and near‑limit mileage
                r = random.random()
                if r < 0.3:
                    # Near limit (for mileage balancing tests)
                    current_mileage = int(max_mileage * random.uniform(0.85, 0.97))
                elif r < 0.6:
                    # Low mileage
                    current_mileage = int(max_mileage * random.uniform(0.25, 0.45))
                else:
                    # Moderate mileage
                    current_mileage = int(max_mileage * random.uniform(0.45, 0.7))
            elif group == "B":
                # Standby: generally mid‑range mileage
                current_mileage = int(max_mileage * random.uniform(0.4, 0.8))
            else:
                # Critical: allow some to be at / above limit (will be caught by safety gate)
                if random.random() < 0.5:
                    current_mileage = int(max_mileage * random.uniform(0.95, 1.05))
                else:
                    current_mileage = int(max_mileage * random.uniform(0.4, 0.9))

            # ---------- Branding (Tier 2) ----------
            advertiser = random.choice(
                [
                    "Kerala Tourism",
                    "Cochin International Airport",
                    "Federal Bank",
                    "None",
                    "MRF Tyres",
                    "Kalyan Jewellers",
                ]
            )

            branding_priority = random.choice(["HIGH", "MEDIUM", "LOW"])
            branding_status = "OPTIONAL"

            if group == "A":
                # About 30% of Golden Fleet have REQUIRED branding with high priority
                if advertiser != "None" and random.random() < 0.3:
                    branding_status = "REQUIRED"
                    branding_priority = "HIGH"

            branding = {
                "current_advertiser": advertiser,
                "contract_expiry": (
                    datetime.now() + timedelta(days=random.randint(30, 180))
                ).isoformat(),
                "priority": branding_priority,
                "revenue_per_day": random.uniform(5000, 15000),
                # Extra flag used for testing Tier 2 behaviour (not mandatory elsewhere)
                "branding_status": branding_status,
            }

            # ---------- Overall status ----------
            if group == "A":
                status = random.choice(["ACTIVE", "STANDBY"])
            elif group == "B":
                status = "STANDBY"
            else:
                status = "MAINTENANCE"

            trainset = {
                "trainset_id": trainset_id,
                "manufacturer": "Alstom",
                "model": "Kochi Metro Coach",
                "year_of_manufacture": 2015 + (i // 5),
                "status": status,
                "current_mileage": current_mileage,
                "max_mileage_before_maintenance": max_mileage,
                "fitness_certificates": fitness_certs,
                "job_cards": job_cards,
                "branding": branding,
                "current_location": {
                    "depot": depot,
                    "bay": f"{depot}_BAY_{bay_num:02d}",
                },
                "last_updated": datetime.now().isoformat(),
            }

            trainsets.append(trainset)

        return trainsets

    async def generate_maximo_job_cards(self):
        """Generate IBM Maximo job cards"""
        job_cards = []
        
        for trainset_id in random.choices(self.trainset_ids, k=50):  # 50 job cards
            job_card = {
                "job_card_id": f"WO{random.randint(100000, 999999)}",
                "trainset_id": trainset_id,
                "work_order_type": random.choice(["PM", "CM", "INSPECTION", "UPGRADE"]),
                "priority": random.choice(["LOW", "NORMAL", "HIGH", "CRITICAL"]),
                "status": random.choice(["OPEN", "IN_PROGRESS", "WAITING_PARTS", "COMPLETED"]),
                "description": random.choice([
                    "Brake pad replacement required",
                    "HVAC system maintenance",
                    "Door mechanism inspection",
                    "Pantograph adjustment",
                    "Bogie bearing lubrication",
                    "Traction motor inspection",
                    "Interior cleaning and sanitization",
                    "External body repair"
                ]),
                "created_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "estimated_duration_hours": random.randint(2, 24),
                "assigned_technician": f"TECH_{random.randint(100, 999)}",
                "estimated_cost": random.uniform(5000, 50000),
                "last_updated": datetime.now().isoformat()
            }
            job_cards.append(job_card)
        
        return job_cards

    async def generate_branding_contracts(self):
        """Generate branding contract records"""
        advertisers = [
            "Kerala Tourism",
            "Cochin International Airport",
            "Federal Bank",
            "MRF Tyres",
            "Kalyan Jewellers",
            "Asian Paints",
            "Reliance Trends",
        ]
        contracts = []
        for adv in advertisers:
            contracts.append({
                "contract_id": f"ADV_{uuid.uuid4().hex[:8].upper()}",
                "advertiser": adv,
                "priority": random.choice(["HIGH", "MEDIUM", "LOW"]),
                "min_daily_exposure_hours": random.choice([6, 8, 10, 12]),
                "start_date": (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%Y-%m-%d'),
                "end_date": (datetime.now() + timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d'),
                "wrap_type": random.choice(["FULL", "PARTIAL", "INTERIOR"]),
                "penalty_per_shortfall_hour": random.randint(2000, 8000),
            })
        return contracts

    async def generate_iot_sensor_data(self):
        """Generate IoT sensor streams data"""
        sensor_data = []
        
        for trainset_id in self.trainset_ids:
            for sensor_type in self.sensor_types:
                # Generate multiple readings over time
                for hours_back in range(0, 72, 2):  # Every 2 hours for 3 days
                    timestamp = datetime.now() - timedelta(hours=hours_back)
                    
                    # Realistic sensor values based on type
                    if sensor_type == "bogie_monitoring":
                        health_score = random.uniform(0.7, 0.98)
                        temperature = random.uniform(25, 45)
                    elif sensor_type == "brake_system":
                        health_score = random.uniform(0.75, 0.95)
                        temperature = random.uniform(30, 80)
                    elif sensor_type == "hvac_control":
                        health_score = random.uniform(0.8, 0.99)
                        temperature = random.uniform(22, 28)
                    else:
                        health_score = random.uniform(0.6, 0.95)
                        temperature = random.uniform(20, 40)
                    
                    sensor_reading = {
                        "trainset_id": trainset_id,
                        "sensor_type": sensor_type,
                        "sensor_id": f"{trainset_id}_{sensor_type}_{random.randint(100, 999)}",
                        "health_score": round(health_score, 2),
                        "temperature": round(temperature, 1),
                        "status": "NORMAL" if health_score > 0.7 else "WARNING" if health_score > 0.5 else "CRITICAL",
                        "timestamp": timestamp.isoformat(),
                    }
                    
                    sensor_data.append(sensor_reading)
        
        return sensor_data

    async def generate_cleaning_schedule(self):
        """Generate cleaning & detailing slots"""
        cleaning_slots = []
        
        for day_offset in range(0, 14):  # Next 14 days
            schedule_date = datetime.now() + timedelta(days=day_offset)
            
            # Generate slots for each depot
            for depot_name, depot_info in self.depots.items():
                cleaning_bays = depot_info["cleaning_bays"]
                
                for bay in range(1, cleaning_bays + 1):
                    # 3 shifts per day
                    for shift in ["MORNING", "AFTERNOON", "NIGHT"]:
                        slot_id = f"CLN_{depot_name}_{schedule_date.strftime('%Y%m%d')}_{shift}_{bay}"
                        
                        cleaning_slot = {
                            "slot_id": slot_id,
                            "date": schedule_date.strftime('%Y-%m-%d'),
                            "shift": shift,
                            "depot": depot_name,
                            "bay_id": f"{depot_name}_CLEANING_BAY_{bay:02d}",
                            "assigned_trainset": random.choice(self.trainset_ids) if random.random() > 0.4 else None,
                            "cleaning_type": random.choice(["BASIC", "DEEP", "MAINTENANCE", "SANITIZATION"]),
                            "duration_hours": random.choice([2, 4, 6, 8]),
                            "crew_assigned": f"CREW_{random.randint(10, 99)}",
                            "status": random.choice(["SCHEDULED", "IN_PROGRESS", "COMPLETED", "CANCELLED"]),
                            "cost_estimate": random.uniform(2000, 8000),
                            "created_at": datetime.now().isoformat()
                        }
                        
                        cleaning_slots.append(cleaning_slot)
        
        return cleaning_slots

    async def generate_historical_operations(self):
        """Generate historical operations data for ML training"""
        historical_data = []
        
        # Generate 90 days of historical induction decisions
        for days_back in range(1, 91):
            operation_date = datetime.now() - timedelta(days=days_back)
            
            for trainset_id in self.trainset_ids:
                # Simulate induction decisions
                decision = random.choices(
                    ["INDUCT", "MAINTENANCE", "STANDBY"],
                    weights=[0.4, 0.3, 0.3]
                )[0]
                
                historical_record = {
                    "operation_date": operation_date.strftime('%Y-%m-%d'),
                    "trainset_id": trainset_id,
                    "induction_status": decision,
                    "actual_service_hours": random.uniform(12, 16) if decision == "INDUCT" else 0,
                    "passenger_count": random.randint(15000, 45000) if decision == "INDUCT" else 0,
                    "revenue_generated": random.uniform(50000, 150000) if decision == "INDUCT" else 0,
                    "energy_consumed_kwh": random.uniform(800, 1200) if decision == "INDUCT" else 0,
                    "maintenance_cost": random.uniform(5000, 25000) if decision == "MAINTENANCE" else 0,
                    "fitness_score_at_induction": random.uniform(0.6, 1.0),
                    "weather_condition": random.choice(["CLEAR", "RAINY", "CLOUDY"]),
                    "peak_hours_operation": random.choice([True, False]),
                    "delayed_minutes": random.randint(0, 15),
                    "customer_complaints": random.randint(0, 3),
                    "ml_prediction_accuracy": random.uniform(0.75, 0.98),
                    "optimization_engine_version": "1.0.0",
                    "created_at": operation_date.isoformat()
                }
                
                historical_data.append(historical_record)
        
        return historical_data

    async def generate_depot_layout_geojson(self):
        """Generate GeoJSON depot layout"""
        geojson_features = []
        
        for depot_name, depot_info in self.depots.items():
            x, y = depot_info["coordinates"]
            delta = 0.002
            depot_feature = {
                "type": "Feature",
                "properties": {
                    "depot_id": f"DEPOT_{depot_name.upper()}",
                    "name": f"{depot_name} Depot",
                    "type": "depot_boundary",
                    "total_bays": depot_info["total_bays"],
                    "maintenance_bays": depot_info["maintenance_bays"],
                    "cleaning_bays": depot_info["cleaning_bays"],
                    "operational_status": "ACTIVE"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [x - delta, y - delta],
                        [x + delta, y - delta],
                        [x + delta, y + delta],
                        [x - delta, y + delta],
                        [x - delta, y - delta]
                    ]]
                }
            }
            geojson_features.append(depot_feature)
        
        depot_geojson = {
            "type": "FeatureCollection",
            "features": geojson_features,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "system": "KMRL Train Induction Planning",
                "version": "1.0",
                "total_depots": len(self.depots),
                "total_bays": sum(d["total_bays"] for d in self.depots.values())
            }
        }
        
        return depot_geojson

# Data Loader following UNS architecture
class KMRLDataLoader:
    """Load mock data into cloud services following UNS architecture"""
    
    def __init__(self):
        self.generator = KMRLMockDataGenerator()
    
    async def load_all_data(self):
        """Load all mock data following the technical architecture data flow"""
        print("Loading KMRL Mock Data into Cloud Services...")
        
        # Generate all data
        mock_data = await self.generator.generate_all_mock_data()
        
        # Connect to cloud services
        await cloud_db_manager.connect_all()
        
        # Load data following UNS pattern:
        
        # 1. MongoDB Atlas (Primary Data Storage)
        await self._load_mongodb_data(mock_data)
        
        # 2. InfluxDB Cloud (Time-series IoT data)
        await self._load_influxdb_data(mock_data["sensor_data"])
        
        # 3. Redis Cloud (Cache frequently accessed data)
        await self._cache_critical_data(mock_data)
        
        print("All mock data loaded successfully!")
        return mock_data
    
    async def _load_mongodb_data(self, mock_data):
        """Load structured data into MongoDB Atlas"""
        collections_data = {
            "trainsets": mock_data["trainsets"],
            "job_cards": mock_data["job_cards"],
            "branding_contracts": mock_data["branding_records"],
            "cleaning_schedule": mock_data["cleaning_schedule"],
            "historical_operations": mock_data["historical_operations"],
            "depot_layout": [mock_data["depot_layout"]]  # Store as single document
        }
        
        for collection_name, data in collections_data.items():
            try:
                collection = await cloud_db_manager.get_collection(collection_name)
                
                # Clear existing data
                await collection.delete_many({})
                
                # Insert new data
                if data:
                    if isinstance(data, list):
                        await collection.insert_many(data)
                    else:
                        await collection.insert_one(data)
                    
                    print(f"   MongoDB: Loaded {len(data) if isinstance(data, list) else 1} records into {collection_name}")
                
            except Exception as e:
                print(f"   MongoDB error loading {collection_name}: {e}")
    
    async def _load_influxdb_data(self, sensor_data):
        """Load time-series sensor data into InfluxDB Cloud"""
        try:
            loaded_count = 0
            
            for sensor_reading in sensor_data:
                success = await cloud_db_manager.write_sensor_data(sensor_reading)
                if success:
                    loaded_count += 1
            
            print(f"   InfluxDB: Loaded {loaded_count} sensor readings")
            
        except Exception as e:
            print(f"   InfluxDB error: {e}")
    
    async def _cache_critical_data(self, mock_data):
        """Cache critical data in Redis Cloud"""
        try:
            # Cache active trainsets
            active_trainsets = [t for t in mock_data["trainsets"] if t["status"] == "ACTIVE"]
            await cloud_db_manager.cache_set("active_trainsets", json.dumps(active_trainsets), expiry=3600)
            
            # Cache depot information
            depot_info = mock_data["depot_layout"]["metadata"]
            await cloud_db_manager.cache_set("depot_info", json.dumps(depot_info), expiry=86400)
            
            print(f"   Redis: Cached critical data")
            
        except Exception as e:
            print(f"   Redis caching error: {e}")

# Main execution function
async def load_kmrl_mock_data():
    """Main function to load all KMRL mock data"""
    try:
        loader = KMRLDataLoader()
        mock_data = await loader.load_all_data()
        
        print("\nMock Data Summary:")
        print(f"   • Trainsets: {len(mock_data['trainsets'])}")
        print(f"   • Job Cards: {len(mock_data['job_cards'])}")
        print(f"   • Sensor Readings: {len(mock_data['sensor_data'])}")
        print(f"   • Cleaning Slots: {len(mock_data['cleaning_schedule'])}")
        print(f"   • Historical Records: {len(mock_data['historical_operations'])}")
        print(f"   • Depot Features: {len(mock_data['depot_layout']['features'])}")
        
        return mock_data
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise
