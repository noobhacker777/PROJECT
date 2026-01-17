class Events:

    def __init__(self) -> None:
        self.enabled = False

    def __call__(self, cfg, device=None) -> None:

events = Events()