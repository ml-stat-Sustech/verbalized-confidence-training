def shutdown_train_dataloader(trainer):
    iterator = getattr(trainer, "train_dataloader_it", None)
    if iterator is None:
        return
    try:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
    finally:
        trainer.train_dataloader_it = None


def finish_swanlab(tracking):
    swanlab = getattr(tracking, "logger", {}).get("swanlab")
    if swanlab is None:
        return
    swanlab.finish()
    tracking.logger.pop("swanlab", None)
