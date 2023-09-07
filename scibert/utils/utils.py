import time

from lightning.pytorch.callbacks import Callback


class TimingCallback(Callback):
    def on_train_start(self) -> None:
        print(f"Training is starting...")
        self.start = time.time()

    def on_train_end(self):
        self.end = time.time()
        total_minutes = (self.end - self.start) / 60
        print(f"Training has finished. It took {total_minutes} min.")
