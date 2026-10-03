#!/usr/bin/env python3
"""Measure a child command using Linux/Python resource counters."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
if not a.command:
    p.error("a command is required")
start = time.monotonic()
result = subprocess.run(a.command)
elapsed = time.monotonic() - start
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
record = {"wall_seconds": elapsed, "exit_code": result.returncode,
          "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
          "cpu_percent": 100 * (usage.ru_utime + usage.ru_stime) / elapsed,
          "max_process_rss_bytes": usage.ru_maxrss * 1024,
          "major_page_faults": usage.ru_majflt, "minor_page_faults": usage.ru_minflt,
          "filesystem_inputs": usage.ru_inblock, "filesystem_outputs": usage.ru_oublock,
          "method": "fresh helper process; Linux resource.getrusage(RUSAGE_CHILDREN), max RSS in KiB converted to bytes"}
a.output.write_text(json.dumps(record, indent=2) + "\n")
raise SystemExit(result.returncode)
