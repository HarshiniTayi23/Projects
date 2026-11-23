import pytest

from data.seed import seed
from mcp_server import db


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    seed(connection, n=75, with_bookings=False)
    yield connection
    connection.close()
