import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

class _AsyncCursor:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

class _MockCollection:
    def __init__(self, items):
        self._items = list(items)

    def find(self, query: Dict[str, Any] | None = None):
        if not query:
            return _AsyncCursor(self._items)
        # rudimentary filtering for tests
        def match(doc):
            for k, v in query.items():
                parts = k.split(".")
                d = doc
                for p in parts:
                    d = d.get(p, {}) if isinstance(d, dict) else None
                if d != v:
                    return False
            return True

        return _AsyncCursor([d for d in self._items if match(d)])

    async def find_one(self, query: Dict[str, Any]):
        async for d in self.find(query):
            return d
        return None

    async def update_one(self, query: Dict[str, Any], payload: Dict[str, Any]):
        # very simple update implementation
        for d in self._items:
            ok = True
            for k, v in query.items():
                if d.get(k) != v:
                    ok = False
                    break
            if ok:
                if "$set" in payload:
                    d.update(payload["$set"]) 
                return type("_Res", (), {"modified_count": 1})()
        return type("_Res", (), {"modified_count": 0})()

    async def insert_one(self, record: Dict[str, Any]):
        self._items.append(dict(record))

    async def insert_many(self, records: List[Dict[str, Any]]):
        for r in records:
            await self.insert_one(r)

    async def delete_many(self, _query: Dict[str, Any]):
        # For mock, clear all when {} provided
        if not _query:
            deleted = len(self._items)
            self._items.clear()
        else:
            # naive filter remove
            keep: List[Dict[str, Any]] = []
            for d in self._items:
                match = True
                for k, v in _query.items():
                    if d.get(k) != v:
                        match = False
                        break
                if not match:
                    keep.append(d)
            deleted = len(self._items) - len(keep)
            self._items = keep
        return type("_DelRes", (), {"deleted_count": deleted})()

class _MockCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}

    async def get(self, key: str):
        await asyncio.sleep(0)
        return self._cache.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        await asyncio.sleep(0)
        self._cache[key] = value

class CloudDatabaseManager:
    """Thin abstraction that reads mock JSON files and provides async-like APIs.

    This allows local development without real MongoDB/Redis/InfluxDB.
    """

    def __init__(self):
        self._root = Path(__file__).resolve().parents[3] / "data" / "mock"
        self._collections: Dict[str, _MockCollection] = {}
        self._cache = _MockCache()

        # preload trainsets if available
        trainsets_path = self._root / "trainsets.json"
        if trainsets_path.exists():
            data = json.loads(trainsets_path.read_text())
        else:
            data = []
        self._collections["trainsets"] = _MockCollection(data)
        # initialize other expected collections
        for name in [
            "optimization_history",
            "job_cards",
            "branding_contracts",
            "cleaning_schedule",
            "historical_operations",
            "depot_layout",
        ]:
            self._collections[name] = _MockCollection([])

    async def get_collection(self, name: str):
        # create on demand if missing
        if name not in self._collections:
            self._collections[name] = _MockCollection([])
        return self._collections[name]

    async def cache_get(self, key: str):
        return await self._cache.get(key)

    async def cache_set(self, key: str, value: str, expiry: int = 300):
        await self._cache.set(key, value, ex=expiry)

    async def write_sensor_data(self, data: Dict[str, Any]):
        # No-op for mock; in real impl, write to InfluxDB
        await asyncio.sleep(0)
        return True

    async def connect_all(self):
        # No real connections needed in mock
        await asyncio.sleep(0)

    async def close_all(self):
        await asyncio.sleep(0)

cloud_db_manager = CloudDatabaseManager()
