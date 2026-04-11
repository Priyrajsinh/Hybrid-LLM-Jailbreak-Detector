def run_redteam(config: dict) -> None:  # type: ignore[type-arg]
    pass


if __name__ == "__main__":
    from src.config import load_config

    run_redteam(load_config())
