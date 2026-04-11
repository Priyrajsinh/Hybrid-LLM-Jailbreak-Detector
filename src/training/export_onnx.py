def export_to_onnx(model_path: str, output_path: str) -> None:
    """Export Stage A to ONNX format via optimum."""
    pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: export_onnx.py <model_path> <output_path>")
        sys.exit(1)
    export_to_onnx(sys.argv[1], sys.argv[2])
