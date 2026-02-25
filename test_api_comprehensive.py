#!/usr/bin/env python3
"""
Comprehensive API endpoint testing script.
Tests all endpoints and verifies data flow through DB and Redis.
"""

import asyncio
import httpx
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8003"
AUTH_TOKEN = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDAiLCJnb29nbHlfdGllciI6ImZyZWUifQ.wqUTdfKu1gZzveanQiQaFUOD-GedGbRtyIoGojHPkA4"

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

test_results = []


def log_test(name: str, status: str, details: str = ""):
    """Log test result with color coding."""
    color = GREEN if status == "PASS" else RED if status == "FAIL" else YELLOW
    print(f"{color}[{status}]{RESET} {name}")
    if details:
        print(f"      {details}")
    test_results.append({"name": name, "status": status, "details": details})


async def test_health_endpoint(client: httpx.AsyncClient):
    """Test /health endpoint."""
    try:
        response = await client.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            if (
                data.get("status") == "healthy"
                and data.get("db") == "ok"
                and data.get("redis") == "ok"
            ):
                log_test("GET /health", "PASS", f"Status: {data}")
                return True
            else:
                log_test("GET /health", "FAIL", f"Unhealthy: {data}")
                return False
        else:
            log_test("GET /health", "FAIL", f"Status code: {response.status_code}")
            return False
    except Exception as e:
        log_test("GET /health", "FAIL", f"Exception: {str(e)}")
        return False


async def test_users_me(client: httpx.AsyncClient):
    """Test GET /api/v1/users/me."""
    try:
        response = await client.get(
            f"{BASE_URL}/api/v1/users/me", headers={"Authorization": AUTH_TOKEN}
        )
        if response.status_code == 200:
            data = response.json()
            log_test(
                "GET /api/v1/users/me", "PASS", f"User ID: {data.get('id', 'N/A')[:20]}"
            )
            return data
        else:
            log_test(
                "GET /api/v1/users/me",
                "FAIL",
                f"Status: {response.status_code}, Body: {response.text[:100]}",
            )
            return None
    except Exception as e:
        log_test("GET /api/v1/users/me", "FAIL", f"Exception: {str(e)}")
        return None


async def test_create_project(client: httpx.AsyncClient):
    """Test POST /api/v1/projects/."""
    try:
        payload = {
            "title": f"Test Project {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "Automated test project for comprehensive API validation and data flow verification",
            "refresh_strategy": "once",
        }
        response = await client.post(
            f"{BASE_URL}/api/v1/projects/",
            headers={"Authorization": AUTH_TOKEN},
            json=payload,
            timeout=60.0,  # Increased timeout for project creation
        )
        if response.status_code == 201:
            data = response.json()
            log_test(
                "POST /api/v1/projects/", "PASS", f"Created project: {data.get('id')}"
            )
            return data
        else:
            log_test(
                "POST /api/v1/projects/",
                "FAIL",
                f"Status: {response.status_code}, Body: {response.text[:200]}",
            )
            return None
    except Exception as e:
        log_test("POST /api/v1/projects/", "FAIL", f"Exception: {str(e)}")
        return None


async def test_list_projects(client: httpx.AsyncClient):
    """Test GET /api/v1/projects/."""
    try:
        response = await client.get(
            f"{BASE_URL}/api/v1/projects/", headers={"Authorization": AUTH_TOKEN}
        )
        if response.status_code == 200:
            data = response.json()
            log_test("GET /api/v1/projects/", "PASS", f"Found {len(data)} projects")
            return data
        else:
            log_test("GET /api/v1/projects/", "FAIL", f"Status: {response.status_code}")
            return None
    except Exception as e:
        log_test("GET /api/v1/projects/", "FAIL", f"Exception: {str(e)}")
        return None


async def test_get_project(client: httpx.AsyncClient, project_id: str):
    """Test GET /api/v1/projects/{project_id}."""
    try:
        response = await client.get(
            f"{BASE_URL}/api/v1/projects/{project_id}",
            headers={"Authorization": AUTH_TOKEN},
        )
        if response.status_code == 200:
            data = response.json()
            log_test(
                f"GET /api/v1/projects/{project_id[:8]}...",
                "PASS",
                f"Name: {data.get('name')}",
            )
            return data
        else:
            log_test(
                f"GET /api/v1/projects/{project_id[:8]}...",
                "FAIL",
                f"Status: {response.status_code}",
            )
            return None
    except Exception as e:
        log_test(
            f"GET /api/v1/projects/{project_id[:8]}...", "FAIL", f"Exception: {str(e)}"
        )
        return None


async def test_chat_history(client: httpx.AsyncClient, project_id: str, user_id: str):
    """Test GET /api/v1/chat/history/{project_id}."""
    try:
        # Generate session_id the same way the API does
        session_id = f"session_{user_id}_{project_id}"
        response = await client.get(
            f"{BASE_URL}/api/v1/chat/history/{project_id}",
            headers={"Authorization": AUTH_TOKEN},
            params={"session_id": session_id},
        )
        if response.status_code == 200:
            data = response.json()
            log_test(
                f"GET /api/v1/chat/history/{project_id[:8]}...",
                "PASS",
                f"Messages: {len(data)}",
            )
            return data
        else:
            log_test(
                f"GET /api/v1/chat/history/{project_id[:8]}...",
                "FAIL",
                f"Status: {response.status_code}, Body: {response.text[:200]}",
            )
            return None
    except Exception as e:
        log_test(
            f"GET /api/v1/chat/history/{project_id[:8]}...",
            "FAIL",
            f"Exception: {str(e)}",
        )
        return None


async def test_chat_streaming(client: httpx.AsyncClient, project_id: str):
    """Test POST /api/v1/chat/ (streaming endpoint)."""
    try:
        payload = {
            "query": "What are the latest beauty trends?",
            "project_id": project_id,
        }

        # Use streaming for SSE
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/v1/chat/",
            headers={"Authorization": AUTH_TOKEN},
            json=payload,
            timeout=30.0,
        ) as response:
            if response.status_code == 200:
                chunks_received = 0
                async for chunk in response.aiter_text():
                    chunks_received += 1
                    if chunks_received >= 3:  # Just verify we get some chunks
                        break

                log_test(
                    f"POST /api/v1/chat/ (stream)",
                    "PASS",
                    f"Received {chunks_received} chunks",
                )
                return True
            else:
                log_test(
                    f"POST /api/v1/chat/ (stream)",
                    "FAIL",
                    f"Status: {response.status_code}",
                )
                return False
    except Exception as e:
        log_test(f"POST /api/v1/chat/ (stream)", "FAIL", f"Exception: {str(e)}")
        return False


async def test_kb_upload_status(client: httpx.AsyncClient, project_id: str):
    """Test GET /api/v1/kb/{project_id}/status."""
    try:
        response = await client.get(
            f"{BASE_URL}/api/v1/kb/{project_id}/status",
            headers={"Authorization": AUTH_TOKEN},
        )
        if response.status_code == 200:
            data = response.json()
            log_test(
                f"GET /api/v1/kb/{project_id[:8]}.../status",
                "PASS",
                f"Chunks: {data.get('kb_chunk_count', 0)}",
            )
            return data
        else:
            log_test(
                f"GET /api/v1/kb/{project_id[:8]}.../status",
                "FAIL",
                f"Status: {response.status_code}",
            )
            return None
    except Exception as e:
        log_test(
            f"GET /api/v1/kb/{project_id[:8]}.../status", "FAIL", f"Exception: {str(e)}"
        )
        return None


async def main():
    """Run all tests."""
    print(f"\n{'=' * 60}")
    print(f"GOOGLY API COMPREHENSIVE ENDPOINT TEST")
    print(f"{'=' * 60}\n")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test 1: Health check
        print("\n--- Infrastructure Tests ---")
        if not await test_health_endpoint(client):
            print(f"\n{RED}CRITICAL: Health check failed. Stopping tests.{RESET}")
            return

        # Test 2: Authentication
        print("\n--- Authentication Tests ---")
        user_data = await test_users_me(client)
        if not user_data:
            print(
                f"\n{YELLOW}WARNING: Authentication failed. Some tests will be skipped.{RESET}"
            )

        # Test 3: Projects
        print("\n--- Project Management Tests ---")
        projects = await test_list_projects(client)

        # Create a new project for testing
        new_project = await test_create_project(client)

        if new_project:
            project_id = new_project.get("id")

            # Get the project we just created
            await test_get_project(client, project_id)

            # Test 4: KB endpoints
            print("\n--- Knowledge Base Tests ---")
            await test_kb_upload_status(client, project_id)

            # Test 5: Chat endpoints
            print("\n--- Chat Tests ---")
            user_id = (
                user_data.get("id")
                if user_data
                else "00000000-0000-0000-0000-000000000000"
            )
            await test_chat_history(client, project_id, user_id)
            await test_chat_streaming(client, project_id)
        else:
            print(
                f"\n{YELLOW}WARNING: Could not create test project. Skipping dependent tests.{RESET}"
            )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 60}\n")

    passed = sum(1 for r in test_results if r["status"] == "PASS")
    failed = sum(1 for r in test_results if r["status"] == "FAIL")
    total = len(test_results)

    print(f"Total Tests: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"Success Rate: {(passed / total * 100):.1f}%\n")

    return failed == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
        sys.exit(1)
