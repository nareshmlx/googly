import asyncio
import pytest
from tests.unit.test_insights_dataflow_guards import test_stream_full_report_scopes_repo_writes_by_project

if __name__ == "__main__":
    pytest.main(["-v", "-s", "tests/unit/test_insights_dataflow_guards.py::test_stream_full_report_scopes_repo_writes_by_project"])
