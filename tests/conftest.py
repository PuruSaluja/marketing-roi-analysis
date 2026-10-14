"""Shared fixtures.

Database tests run against ``MARKETING_ROI_TEST_DSN`` and drop and recreate the
``marketing`` and ``truth`` schemas there, so the run is refused unless the database's
name ends in ``_test``. Without the variable they are skipped, unless
``MARKETING_ROI_REQUIRE_DB`` is set (as CI sets it), which turns the skip into a failure
so a missing database cannot pass unnoticed.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

TEST_DSN = os.environ.get("MARKETING_ROI_TEST_DSN", "")


@pytest.fixture(scope="session")
def test_dsn() -> str:
    if not TEST_DSN:
        if os.environ.get("MARKETING_ROI_REQUIRE_DB"):
            pytest.fail("MARKETING_ROI_REQUIRE_DB is set but MARKETING_ROI_TEST_DSN is not")
        pytest.skip("set MARKETING_ROI_TEST_DSN to run the database tests")
    dbname = conninfo_to_dict(TEST_DSN).get("dbname", "")
    if not dbname.endswith("_test"):
        pytest.fail(
            f"Refusing to run: test database {dbname!r} does not end in '_test'. The database "
            "tests drop the marketing and truth schemas, so point MARKETING_ROI_TEST_DSN at a "
            "disposable database."
        )
    return TEST_DSN


@pytest.fixture
def db(test_dsn: str):
    """A connection to the test database with the schemas dropped, closed after the test."""
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("DROP SCHEMA IF EXISTS marketing, truth CASCADE")
        yield conn
