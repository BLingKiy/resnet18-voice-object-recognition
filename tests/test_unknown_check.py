"""`tools/unknown_check.py` 的拒识统计测试（不加载真实模型）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from helpers import solid
from PIL import Image

from tools import unknown_check


def _write_images(root: Path, files: dict[str, list[str]]) -> Path:
    """在 root/<类名>/ 下写入纯色占位图，返回 root。"""
    for class_name, names in files.items():
        directory = root / class_name
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            solid((10, 10, 10)).save(directory / name)
    return root


def test_summary_reports_rates_and_highest(  ) -> None:
    rows = [
        {"class": "剪刀", "image": "a.jpg", "nearest": "mouse", "similarity": 0.60, "is_known": False, "false_positive": False},
        {"class": "充电头", "image": "b.jpg", "nearest": "power_bank", "similarity": 0.74, "is_known": False, "false_positive": False},
        {"class": "乌龙茶", "image": "c.jpg", "nearest": "water_cup", "similarity": 0.85, "is_known": True, "false_positive": True},
    ]

    summary = unknown_check.summarize(rows, 0.8124)

    assert summary["total"] == 3
    assert summary["rejected"] == 2
    assert summary["false_positive"] == 1
    assert summary["false_positive_rate"] == pytest.approx(1 / 3)
    assert summary["rejection_rate"] == pytest.approx(2 / 3)
    assert summary["highest"]["image"] == "c.jpg"
    assert summary["highest"]["margin"] == pytest.approx(0.8124 - 0.85, abs=1e-6)


def test_collect_rows_flags_false_positive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_images(tmp_path / "unknown", {"充电头": ["1.jpg", "2.jpg"]})

    def fake_recognize(library: Any, image: Any, *, threshold: Any = None, top_k: int = 1) -> Any:
        # 第一张判成已知（误报），第二张判未知（正确拒识）
        known = Path(image).name == "1.jpg"
        return SimpleNamespace(
            is_known=known,
            name="power_bank" if known else "未知物体",
            similarity=0.90 if known else 0.70,
            object_scores=(SimpleNamespace(name="power_bank", max_similarity=0.90),),
        )

    monkeypatch.setattr(unknown_check, "recognize", fake_recognize)

    rows = unknown_check.collect_rows(SimpleNamespace(), root, None)

    assert [row["image"] for row in rows] == ["1.jpg", "2.jpg"]
    assert rows[0]["false_positive"] is True
    assert rows[1]["false_positive"] is False
    assert rows[0]["nearest"] == "power_bank"


def test_main_returns_one_when_false_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_images(tmp_path / "unknown", {"充电头": ["1.jpg"]})
    library_path = tmp_path / "features.json"
    library_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        unknown_check,
        "load_library",
        lambda path=None, create=True: SimpleNamespace(total_vectors=1, threshold=0.8124),
    )
    monkeypatch.setattr(
        unknown_check,
        "recognize",
        lambda library, image, **kwargs: SimpleNamespace(
            is_known=True,
            name="power_bank",
            similarity=0.90,
            object_scores=(SimpleNamespace(name="power_bank", max_similarity=0.9),),
        ),
    )

    code = unknown_check.main(["--library", str(library_path), "--dir", str(root)])

    assert code == 1
    assert "误报 1/1" in capsys.readouterr().out


def test_missing_directory_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    library_path = tmp_path / "features.json"
    library_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        unknown_check,
        "load_library",
        lambda path=None, create=True: SimpleNamespace(total_vectors=1, threshold=0.8124),
    )

    code = unknown_check.main(
        ["--library", str(library_path), "--dir", str(tmp_path / "nope")]
    )

    assert code == 2
    assert "目录不存在" in capsys.readouterr().err
