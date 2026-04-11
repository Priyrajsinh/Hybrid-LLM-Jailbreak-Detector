def evaluate(config: dict) -> None:  # type: ignore[type-arg]
    pass


if __name__ == "__main__":
    from src.config import load_config

    evaluate(load_config())
