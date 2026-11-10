import subprocess
import sys

_WORKER = r'''
import sys
from datetime import timedelta
from pathlib import Path
import torch.distributed as dist
from supermimic.training.distributed import broadcast_setup_error

rank = int(sys.argv[1])
init_file = sys.argv[2]
root = Path(sys.argv[3])
dist.init_process_group(
    "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
    timeout=timedelta(seconds=15),
)
if rank == 0:
    (root / "created_by_rank0").write_text("ready")
error = broadcast_setup_error(None)
dist.barrier()
seen = (root / "created_by_rank0").read_text()
(root / f"success_{rank}").write_text(f"{seen}:{error}")
error = broadcast_setup_error("simulated rank-zero setup failure" if rank == 0 else None)
(root / f"failure_{rank}").write_text(str(error))
dist.destroy_process_group()
'''


def test_rank_zero_setup_is_visible_before_workers_continue(tmp_path):
    init_file = tmp_path / "gloo-init"
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(rank), str(init_file), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    try:
        for proc in processes:
            stdout, stderr = proc.communicate(timeout=30)
            assert proc.returncode == 0, stderr or stdout
    except subprocess.TimeoutExpired:
        for proc in processes:
            proc.kill()
        for proc in processes:
            proc.communicate()
        raise
    for rank in range(2):
        assert (tmp_path / f"success_{rank}").read_text() == "ready:None"
        assert (tmp_path / f"failure_{rank}").read_text() == "simulated rank-zero setup failure"
