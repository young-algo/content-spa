import asyncio

from fastapi import APIRouter, HTTPException

from pci.api.schemas import VaultExportRequest, VaultExportResponse

router = APIRouter(prefix="/api/vault", tags=["vault"])


@router.post("/export", response_model=VaultExportResponse)
async def export_vault(req: VaultExportRequest):
    try:
        from pci.vault import export_vault as _export_vault

        result = await asyncio.to_thread(
            _export_vault,
            vault_dir=req.vault_dir,
            include_content=not req.no_content,
            source_type=req.source_type,
            limit=req.limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return VaultExportResponse(
        exported=result["exported"],
        skipped=result["skipped"],
        vault_dir=result["vault_dir"],
        output_dir=result["output_dir"],
    )
