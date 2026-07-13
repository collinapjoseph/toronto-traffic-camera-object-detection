# Object Detection at Toronto Intersections
Performs object detection on images from City of Toronto intersection traffic cameras using YOLO26.<br>
Allows user selection of intersections and displays results through Gradio UI.<br>
Objects detected: `[car, truck, bus, motorcycle, bicycle, person]`<br>
## Requirements
`pip install requests pillow matplotlib pandas torch ultralytics opencv-python`<br>
## Run
1. `python main.py`
2. Open browser and navigate to [localhost:7860](http://127.0.0.1:7860)

