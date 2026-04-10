from __future__ import annotations

import base64
import json
from pathlib import Path


def export_notebook_png_outputs(notebook_path: Path, output_dir: Path) -> int:
    """Export all image/png cell outputs from a notebook into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    image_count = 0
    for cell_idx, cell in enumerate(cells, start=1):
        outputs = cell.get("outputs", [])
        for output_idx, output in enumerate(outputs, start=1):
            data = output.get("data", {})
            png_data = data.get("image/png")
            if not png_data:
                continue

            if isinstance(png_data, list):
                png_data = "".join(png_data)

            image_bytes = base64.b64decode(png_data)
            image_count += 1
            file_name = f"figure_{image_count:02d}_cell_{cell_idx:02d}_out_{output_idx:02d}.png"
            (output_dir / file_name).write_bytes(image_bytes)

    return image_count


def main() -> None:
    dataset_dir = Path(__file__).resolve().parent
    notebook_path = dataset_dir / "water_quality_analysis.ipynb"
    output_dir = dataset_dir / "smote_graphs"

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    count = export_notebook_png_outputs(notebook_path, output_dir)
    print(f"Saved {count} image(s) to: {output_dir}")


if __name__ == "__main__":
    main()
