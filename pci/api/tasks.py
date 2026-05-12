import asyncio
import uuid
from typing import Any, Callable, Dict, Optional

_tasks: Dict[str, dict] = {}


def create_task() -> str:
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "",
        "result": None,
        "error": None,
    }
    return task_id


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def update_task(task_id: str, **kwargs: Any) -> None:
    if task_id in _tasks:
        _tasks[task_id].update(kwargs)


async def run_task(task_id: str, coro):
    _tasks[task_id]["status"] = "running"
    try:
        result = await coro
        _tasks[task_id]["status"] = "complete"
        _tasks[task_id]["progress"] = 100.0
        _tasks[task_id]["result"] = result
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
