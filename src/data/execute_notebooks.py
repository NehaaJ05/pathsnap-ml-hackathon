"""Execute and save the five Member 1 notebooks using the current interpreter."""
from pathlib import Path
import json
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


def execute(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    results = []
    with tempfile.TemporaryDirectory(prefix="pathsnap-kernel-") as temp:
        spec = Path(temp) / "kernels" / "python3"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "PathSnap verification", "language": "python"}), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[str(spec.parent)])
        for path in sorted((root / "notebooks").glob("0[1-5]_*.ipynb")):
            print(f"Executing {path.name}", flush=True)
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                    resources={"metadata": {"path": str(root / "notebooks")}},
                                    store_widget_state=False)
            # Supply a temporary kernelspec explicitly bound to this interpreter.
            client.create_kernel_manager()
            client.km.kernel_spec_manager = manager
            client.execute()
            nbformat.validate(nb)
            nbformat.write(nb, path)
            code = [c for c in nb.cells if c.cell_type == "code"]
            errors = sum(o.output_type == "error" for c in code for o in c.outputs)
            if errors:
                raise RuntimeError(f"Notebook contains errors: {path.name}")
            results.append({"notebook": path.name, "code_cells": len(code),
                            "executed_cells": sum(c.execution_count is not None for c in code),
                            "errors": errors})
    output = root / "results/member1/notebook_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    execute()
