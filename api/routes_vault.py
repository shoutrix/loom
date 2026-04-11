"""GET/POST /vault -- list/create/edit notes."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/vault", tags=["vault"])


class CreateNoteRequest(BaseModel):
    title: str
    content: str
    subfolder: str = "notes"


class WriteFileRequest(BaseModel):
    relative_path: str
    content: str


class VaultFileResponse(BaseModel):
    relative_path: str
    title: str
    size_bytes: int


@router.get("/files", response_model=list[VaultFileResponse])
async def list_files(subfolder: str = "") -> list[VaultFileResponse]:
    from loom.main import get_app_state
    state = get_app_state()

    files = state.vault.list_files(subfolder)
    return [
        VaultFileResponse(
            relative_path=f.relative_path,
            title=f.title,
            size_bytes=f.size_bytes,
        )
        for f in files
    ]


@router.get("/read")
async def read_file(path: str) -> dict:
    from loom.main import get_app_state
    state = get_app_state()

    content = state.vault.read_file(path)
    if content is None:
        return {"error": "File not found"}
    return {"path": path, "content": content}


@router.post("/note", response_model=VaultFileResponse)
async def create_note(req: CreateNoteRequest) -> VaultFileResponse:
    from loom.main import get_app_state
    state = get_app_state()

    vf = state.vault.create_note(req.title, req.content, req.subfolder)
    return VaultFileResponse(
        relative_path=vf.relative_path,
        title=vf.title,
        size_bytes=vf.size_bytes,
    )


@router.post("/write", response_model=VaultFileResponse)
async def write_file(req: WriteFileRequest) -> VaultFileResponse:
    from loom.main import get_app_state
    state = get_app_state()

    vf = state.vault.write_file(req.relative_path, req.content)
    return VaultFileResponse(
        relative_path=vf.relative_path,
        title=vf.title,
        size_bytes=vf.size_bytes,
    )


@router.delete("/delete")
async def delete_file(path: str) -> dict:
    from loom.main import get_app_state
    state = get_app_state()

    success = state.vault.delete_file(path)
    return {"deleted": success, "path": path}
