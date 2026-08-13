import pytest

from app.repositories.mongo_users import MongoUserRepository


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self.documents):
            raise StopAsyncIteration
        item = self.documents[self._index]
        self._index += 1
        return item


class FakeCollection:
    def find(self, query):
        self.query = query
        return FakeCursor([
            {"_id": "mongo-id", "username": "driver1", "email_verified": True, "is_approved": False, "role": "METRO_DRIVER"},
        ])


class FakeDatabaseManager:
    def __init__(self):
        self.collection = FakeCollection()

    async def get_collection(self, name):
        assert name == "users"
        return self.collection


@pytest.mark.asyncio
async def test_list_pending_removes_sensitive_fields(monkeypatch):
    manager = FakeDatabaseManager()
    monkeypatch.setattr("app.repositories.mongo_users.cloud_db_manager", manager)

    result = await MongoUserRepository().list_pending()
    assert result == [{"username": "driver1", "email_verified": True, "is_approved": False, "role": "METRO_DRIVER"}]
    assert manager.collection.query["is_approved"] is False
