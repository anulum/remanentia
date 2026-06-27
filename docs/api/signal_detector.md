# signal_detector

`signal_detector.py` detects correction and reinforcement signals in user
messages. Knowledge-store ingestion uses these signals to adjust confidence and
preserve explicit corrections as first-class memory events.

```python
from signal_detector import SignalType, classify_message, detect_signals

signals = detect_signals("Correction: that date was wrong.")
assert classify_message("exactly right") is SignalType.REINFORCEMENT
```

::: signal_detector
