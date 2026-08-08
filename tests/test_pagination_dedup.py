import datetime

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas


@pytest.mark.asyncio
async def test_conclusions_list_returns_distinct_ids_across_pages(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, observer = sample_data
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([observed, session])
    await db_session.flush()

    collection = models.Collection(
        workspace_name=workspace.name,
        observer=observer.name,
        observed=observed.name,
    )
    db_session.add(collection)
    await db_session.flush()

    created_at = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    db_session.add_all(
        [
            models.Document(
                workspace_name=workspace.name,
                observer=observer.name,
                observed=observed.name,
                content=f"Conclusion {index}",
                session_name=session.name,
                created_at=created_at + datetime.timedelta(seconds=index),
            )
            for index in range(150)
        ]
    )
    await db_session.commit()

    page_ids: list[set[str]] = []
    for page in (1, 2, 3):
        response = client.post(
            f"/v3/workspaces/{workspace.name}/conclusions/list",
            params={"page": page, "size": 50},
            json={},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 150
        page_ids.append({item["id"] for item in payload["items"]})

    assert all(len(ids) == 50 for ids in page_ids)
    assert page_ids[0].isdisjoint(page_ids[1])
    assert page_ids[0].isdisjoint(page_ids[2])
    assert page_ids[1].isdisjoint(page_ids[2])
    assert len(set().union(*page_ids)) == 150


@pytest.mark.asyncio
async def test_create_observations_deduplicates_repeated_batch(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> None:
    workspace, observer = sample_data
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    session = models.Session(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([observed, session])
    await db_session.commit()

    observation = schemas.ConclusionCreate(
        content="The same observation",
        observer_id=observer.name,
        observed_id=observed.name,
        session_id=session.name,
    )

    initially_created = await crud.create_observations(
        db_session,
        observations=[observation],
        workspace_name=workspace.name,
    )
    duplicates_created = await crud.create_observations(
        db_session,
        observations=[observation] * 99,
        workspace_name=workspace.name,
    )

    row_count = await db_session.scalar(
        select(func.count())
        .select_from(models.Document)
        .where(
            models.Document.workspace_name == workspace.name,
            models.Document.observer == observer.name,
            models.Document.observed == observed.name,
            models.Document.session_name == session.name,
        )
    )
    assert len(initially_created) == 1
    assert duplicates_created == []
    assert row_count == 1
