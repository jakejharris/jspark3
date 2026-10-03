"""Idle followers wait on the rendezvous socket before entering the unchanged NCCL protocol."""

from datetime import timedelta
import os


def configured() -> bool:
    value = os.environ.get("TF_GLM_FOLLOWER_DOORBELL", "0")
    if value not in ("0", "1"):
        raise ValueError("TF_GLM_FOLLOWER_DOORBELL must be 0 or 1")
    return value == "1"


class FollowerDoorbell:
    """One consumed key per follower: a fast rank cannot delete another rank's wake-up."""

    def __init__(self, store, rank: int, world: int):
        self.store, self.rank, self.world = store, rank, world
        self.epoch = 0

    def ring(self) -> None:
        self.epoch += 1
        for rank in range(1, self.world):
            self.store.set(f"tf_glm_idle_{self.epoch}_{rank}", b"1")

    def wait(self) -> None:
        from torch.distributed import DistStoreError

        key = f"tf_glm_idle_{self.epoch + 1}_{self.rank}"
        while True:
            try:
                self.store.wait([key], timedelta(hours=1))
                break
            except DistStoreError as exc:
                # An idle hour is harmless; connection loss and other errors must escape.
                if "timeout" not in str(exc).lower():
                    raise
        self.store.delete_key(key)
        self.epoch += 1
