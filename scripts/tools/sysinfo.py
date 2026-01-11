#Projet PEPM By Yi Fan && Adrien
#!/usr/bin/env python3
# scripts/sysinfo.py
from __future__ import annotations
import json
import os
import platform
import shutil
from pathlib import Path


def read_meminfo() -> dict:
    info: dict[str, str] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
    except OSError:
        pass
    return info


def fmt_bytes_to_gb(value: float) -> str:
    try:
        return f"{value / (1024 ** 3):.2f} GB"
    except Exception:
        return "?"


def main() -> None:
    print("=== SYSTEM INFORMATION ===")
    print("[OS ]", platform.platform())
    print("[PY ]", platform.python_version())
    print("[CPU]", os.cpu_count(), "cores available")

    env_threads = {
        "OMP": os.getenv("OMP_NUM_THREADS"),
        "OPENBLAS": os.getenv("OPENBLAS_NUM_THREADS"),
        "MKL": os.getenv("MKL_NUM_THREADS"),
        "NUMEXPR": os.getenv("NUMEXPR_NUM_THREADS"),
    }
    print("[ENV]", env_threads)

    try:
        import psutil

        vm = psutil.virtual_memory()
        print(
            "[RAM] total=%s available=%s used=%d%%"
            % (fmt_bytes_to_gb(vm.total), fmt_bytes_to_gb(vm.available), vm.percent)
        )
    except Exception:
        meminfo = read_meminfo()
        total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        if total and available:
            try:
                total_kb = float(total.split()[0])
                available_kb = float(available.split()[0])
                print(
                    "[RAM] total=%s available=%s (psutil not installed)"
                    % (fmt_bytes_to_gb(total_kb * 1024), fmt_bytes_to_gb(available_kb * 1024))
                )
            except Exception:
                print("[RAM] information unavailable")
        else:
            print("[RAM] information unavailable")

    try:
        usage = shutil.disk_usage(Path.cwd())
        print(
            "[DISK] used=%s total=%s free=%s"
            % (fmt_bytes_to_gb(usage.used), fmt_bytes_to_gb(usage.total), fmt_bytes_to_gb(usage.free))
        )
    except Exception as exc:
        print("[DISK] error:", exc)

    try:
        import spacy

        print("[spaCy]", spacy.__version__)
        nlp = spacy.blank("fr")
        doc = nlp("Test rapide.")
        print("[spaCy] blank model OK:", [token.text for token in doc])
    except Exception as exc:
        print("[spaCy] unavailable:", exc)

    summary = {
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "threads_env": env_threads,
    }
    print("[SUMMARY]", json.dumps(summary))


if __name__ == "__main__":
    main()
