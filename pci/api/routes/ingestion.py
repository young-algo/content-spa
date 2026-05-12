import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from pci.api.schemas import IngestURLRequest, TaskStatus, DocumentOut
from pci.api.tasks import create_task, get_task, run_task
from pci.db import get_document_by_url
from pci.ingest import async_ingest_url, async_ingest_local_file

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


@router.post("/url", response_model=TaskStatus)
async def ingest_url(req: IngestURLRequest, background_tasks: BackgroundTasks):
    task_id = create_task()
    background_tasks.add_task(run_task, task_id, _ingest_url_task(req.url, task_id))
    return TaskStatus.model_validate(get_task(task_id))


async def _ingest_url_task(url: str, task_id: str):
    from pci.api.tasks import update_task

    update_task(task_id, message=f"Ingesting {url}...")
    doc_id = await async_ingest_url(url)

    if doc_id:
        update_task(task_id, message="Ingestion complete")
        return {"doc_id": doc_id}
    else:
        raise RuntimeError("Extraction failed or URL already indexed")


async def _ingest_file_task(file_path: str, task_id: str):
    from pci.api.tasks import update_task

    update_task(task_id, message=f"Ingesting file: {os.path.basename(file_path)}...")
    doc_id = await async_ingest_local_file(file_path)

    try:
        os.unlink(file_path)
    except OSError:
        pass

    if doc_id:
        update_task(task_id, message="Ingestion complete")
        return {"doc_id": doc_id}
    else:
        raise RuntimeError("Extraction failed or file already indexed")


@router.post("/file", response_model=TaskStatus)
async def ingest_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    suffix = os.path.splitext(file.filename or "upload")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    task_id = create_task()
    background_tasks.add_task(run_task, task_id, _ingest_file_task(tmp_path, task_id))
    return TaskStatus.model_validate(get_task(task_id))


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus.model_validate(task)
