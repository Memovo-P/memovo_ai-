"""Shared fixtures for the integration suite."""

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from memovo_ai.main import wait_for_initialization


@pytest.fixture
def settled() -> Callable[[TestClient], None]:
    """Block a synchronous test until the app's background build has finished.

    Startup no longer waits for the services: the lifespan yields while the
    model is still loading, and ``/ready`` reports ``loading`` until the
    background task completes. A test that wants to observe the *outcome*
    of startup calls ``settled(client)`` first, on the client's own portal so
    the wait runs on the application's event loop.
    """

    def wait(client: TestClient) -> None:
        portal = client.portal
        assert portal is not None, "settled() needs a client that has been entered"
        portal.call(wait_for_initialization, client.app)

    return wait
