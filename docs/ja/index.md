# ghandドキュメント

ghandは、4指16自由度のロボットハンド「LEAP V1 KAIZEN」を操作・テレオペレーション・シミュレーションし、学習データを収集するためのPythonツールキットです。

!!! warning "最初は実機を使わずに確認してください"

    ghandは実機のDynamixelアクチュエータを駆動できます。最初はFakeBusまたはMuJoCoを使用し、実機でトルクを有効にする前にハードウェア設定と安全確認を完了してください。

## 目的から探す

| 目的 | ページ |
| --- | --- |
| ロボットなしで試す | [Safe quickstart](getting-started/quickstart.md) |
| 実機を接続する | [Hardware setup](getting-started/hardware-setup.md) |
| Webカメラで操作する | [Teleoperation](guides/teleoperation.md) |
| キャリブレーションする | [Calibration and diagnostics](guides/calibration-and-diagnostics.md) |
| LeRobotデータを収録する | [LeRobot datasets](data-and-learning/lerobot-dataset.md) |
| MuJoCoを使う | [MuJoCo](simulation/mujoco.md) |
| コマンドを調べる | [CLI reference](reference/cli.md) |

現時点では英語版を内容の正本とし、未翻訳ページは英語版を表示します。

## ローカルプレビュー

```bash
uv run --group docs mkdocs serve
```

ブラウザで<http://127.0.0.1:8000/ja/>を開きます。
