import io
import json
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_import_is_silent_and_installs_no_output_handlers():
    script = """
import json
import logging

root_logger = logging.getLogger()
root_handlers_before = list(root_logger.handlers)

import hybrid_dynamics

package_logger = logging.getLogger("hybrid_dynamics")
module_handler_types = {
    name: [type(handler).__name__ for handler in logging.getLogger(name).handlers]
    for name in (
        "hybrid_dynamics.src.cubifier",
        "hybrid_dynamics.src.evaluation",
    )
}
print(json.dumps({
    "root_unchanged": root_logger.handlers == root_handlers_before,
    "package_handler_types": [
        type(handler).__name__ for handler in package_logger.handlers
    ],
    "module_handler_types": module_handler_types,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.stderr == ""
    assert payload == {
        "root_unchanged": True,
        "package_handler_types": ["NullHandler"],
        "module_handler_types": {
            "hybrid_dynamics.src.cubifier": [],
            "hybrid_dynamics.src.evaluation": [],
        },
    }


def test_get_logger_compatibility_does_not_configure_handlers():
    from hybrid_dynamics import config, get_logger

    name = "hybrid_dynamics.tests.compatibility"
    logger = logging.getLogger(name)
    logger.handlers.clear()

    assert config.get_logger(name) is logger
    assert get_logger(name) is logger
    assert logger.handlers == []


def test_configure_logging_is_explicit_and_idempotent():
    from hybrid_dynamics import configure_logging, get_logger

    package_logger = logging.getLogger("hybrid_dynamics")
    original_handlers = list(package_logger.handlers)
    original_level = package_logger.level
    original_propagate = package_logger.propagate
    stream = io.StringIO()

    try:
        configure_logging(level="INFO", stream=stream)
        configure_logging(level="INFO", stream=stream)

        get_logger("hybrid_dynamics.tests.configured").info("configured once")

        assert stream.getvalue().count("configured once") == 1
        output_handlers = [
            handler
            for handler in package_logger.handlers
            if isinstance(handler, (logging.StreamHandler, logging.FileHandler))
            and not isinstance(handler, logging.NullHandler)
        ]
        assert len(output_handlers) == 1
    finally:
        for handler in list(package_logger.handlers):
            if handler not in original_handlers:
                package_logger.removeHandler(handler)
                handler.close()
        package_logger.setLevel(original_level)
        package_logger.propagate = original_propagate


def test_multigrid_failures_are_logged_as_one_summary(caplog):
    import networkx as nx

    from hybrid_dynamics import Grid, MultiGrid

    class FailingSystem:
        def simulate(self, point, time_span):
            raise RuntimeError("sample failed")

    mode_graph = nx.DiGraph()
    mode_graph.add_node(0)
    multigrid = MultiGrid(
        mode_graph,
        {0: Grid(bounds=[[0.0, 1.0]], subdivisions=[1])},
    )

    with caplog.at_level(logging.WARNING, logger="hybrid_dynamics.src.multigrid"):
        box_map = multigrid.compute_multi_boxmap(
            FailingSystem(),
            tau=1.0,
            bloat_factor=0.0,
        )

    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.name == "hybrid_dynamics.src.multigrid"
    ]
    assert box_map == {}
    assert len(summaries) == 1
    assert "skipped 2 item(s)" in summaries[0]
    assert "simulation failure (RuntimeError): 2" in summaries[0]


def test_timing_uses_logging_but_progress_callback_keeps_stdout(caplog, capsys):
    from hybrid_dynamics.src.demo_utils import (
        create_progress_callback,
        timed_operation,
    )

    @timed_operation("Small operation")
    def operation():
        return 7

    with caplog.at_level(logging.INFO, logger="hybrid_dynamics.src.demo_utils"):
        assert operation() == 7

    assert capsys.readouterr().out == ""
    assert any(
        "Small operation computed in" in record.getMessage()
        for record in caplog.records
    )

    progress = create_progress_callback(update_interval=1)
    progress(1, 1)
    assert "Progress: 1/1 (100.0%)" in capsys.readouterr().out
