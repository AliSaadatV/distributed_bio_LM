import os, torch
print(
    "rank", os.environ.get("RANK"),
    "local_rank", os.environ.get("LOCAL_RANK"),
    "cuda", torch.cuda.is_available(),
    "device_count", torch.cuda.device_count()
)
