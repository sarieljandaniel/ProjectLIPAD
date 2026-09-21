# Direct MJPEG camera viewer

The existing `updated_main_app.py` uses the MPEG-TS receiver for live analysis.
The requested OpenCV/PyQt-style HTTP MJPEG display is available through the
CustomTkinter-compatible entry point:

```bash
cd ProjectLIPAD/PROJECT_LIPAD
python mjpeg_main_app.py
```

Open **Inspection Manager**, enter the Raspberry Pi endpoint (for example
`http://192.168.1.50:5000/video_feed`), and select **Start camera feed**.

The viewer decodes frames on a daemon thread and schedules only UI updates on
the Tk main thread. Stopping the feed releases the OpenCV capture. This display
path is independent from the existing live-analysis engine and does not produce
inference output or CSV results.
