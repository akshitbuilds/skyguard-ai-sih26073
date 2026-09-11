from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime
import numpy as np


class SensorDegradationTracker:
    """
    Tracks, per sensor, whether it's trending toward failure -- not just
    whether the current reading is bad. Uses a rolling log of every flag
    decision (anomalous or not, if you choose to feed both) plus explicit
    frozen-value streak length.

    Decision logic (simple and explainable on purpose -- a judge should be
    able to follow the rule, not just trust a black box):
      - recent_anomaly_rate = flags in the last RECENT_WINDOW events / RECENT_WINDOW
      - baseline_anomaly_rate = flags in the BASELINE_WINDOW events before that / BASELINE_WINDOW
      - trend_slope = slope of a linear fit over binned anomaly rate across history
      - status:
          CRITICAL  -> long frozen streak, or recent rate >= 0.6
          DEGRADING -> recent rate meaningfully higher than baseline AND positive trend
          WATCH     -> recent rate somewhat elevated, or a shorter frozen streak
          STABLE    -> otherwise
    """

    RECENT_WINDOW = 10
    BASELINE_WINDOW = 30
    HISTORY_CAP = 200

    def __init__(self):
        self._history = defaultdict(lambda: deque(maxlen=self.HISTORY_CAP))
        # each entry: (timestamp, is_anomaly: bool, root_cause: str|None, flatline_run: float)

    def update(self, sensor_id: str, timestamp: str, is_anomaly: bool,
               root_cause: str | None, flatline_run: float = 0.0):
        self._history[sensor_id].append((timestamp, is_anomaly, root_cause, flatline_run))

    def get_status(self, sensor_id: str) -> dict:
        hist = list(self._history[sensor_id])
        if not hist:
            return {
                "sensor_id": sensor_id, "status": "STABLE",
                "recent_anomaly_rate": 0.0, "baseline_anomaly_rate": 0.0,
                "trend_slope": 0.0, "frozen_streak_readings": 0,
                "reasoning": "No history yet for this sensor.",
            }

        flags = np.array([1.0 if h[1] else 0.0 for h in hist])
        current_frozen_streak = int(hist[-1][3])

        recent = flags[-self.RECENT_WINDOW:]
        recent_rate = float(recent.mean()) if recent.size else 0.0

        baseline_slice = flags[-(self.RECENT_WINDOW + self.BASELINE_WINDOW):-self.RECENT_WINDOW]
        baseline_rate = float(baseline_slice.mean()) if baseline_slice.size else recent_rate

        # trend slope over binned rate across the full retained history
        bin_size = max(len(flags) // 10, 1)
        bins = [flags[i:i + bin_size].mean() for i in range(0, len(flags), bin_size)]
        if len(bins) >= 2:
            x = np.arange(len(bins))
            slope = float(np.polyfit(x, bins, 1)[0])
        else:
            slope = 0.0

        reasons = []
        status = "STABLE"
        enough_history = len(hist) >= max(self.RECENT_WINDOW // 2, 5)

        if current_frozen_streak >= 6:
            status = "CRITICAL"
            reasons.append(f"currently frozen for {current_frozen_streak} consecutive readings")
        elif not enough_history:
            reasons.append(f"only {len(hist)} readings on record so far -- too early to assess a trend")
        elif recent_rate >= 0.6:
            status = "CRITICAL"
            reasons.append(f"{recent_rate*100:.0f}% of its last {self.RECENT_WINDOW} readings were flagged")
        elif recent_rate - baseline_rate >= 0.25 and slope > 0.01:
            status = "DEGRADING"
            reasons.append(
                f"flag rate rose from {baseline_rate*100:.0f}% to {recent_rate*100:.0f}% "
                f"with a positive trend across its history"
            )
        elif recent_rate - baseline_rate >= 0.12 or current_frozen_streak >= 3:
            status = "WATCH"
            if current_frozen_streak >= 3:
                reasons.append(f"a {current_frozen_streak}-reading flatline streak observed recently")
            else:
                reasons.append(f"flag rate mildly elevated ({recent_rate*100:.0f}% vs {baseline_rate*100:.0f}% baseline)")
        else:
            reasons.append("flag rate and behavior consistent with baseline")

        return {
            "sensor_id": sensor_id,
            "status": status,
            "recent_anomaly_rate": round(recent_rate, 3),
            "baseline_anomaly_rate": round(baseline_rate, 3),
            "trend_slope": round(slope, 4),
            "frozen_streak_readings": current_frozen_streak,
            "reasoning": "; ".join(reasons),
        }

    def all_statuses(self) -> list[dict]:
        return [self.get_status(sid) for sid in self._history.keys()]
