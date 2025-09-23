import cv2
from urllib.parse import quote

def extract_rtsp_cap():
    username = quote("admin")
    password = quote("admin@123")
    ip = "192.168.0.101"
    stream = "stream1"  # try "stream2" if this fails

    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/{stream}"
    print(f"Testing RTSP on GST: {rtsp_url}")
    
    # Optimized GStreamer pipeline for lower latency
    gst_pipeline = (
        f"rtspsrc location={rtsp_url} "
        f"latency=200 buffer-mode=1 protocols=tcp ! "
        f"rtph264depay ! h264parse ! "
        f"avdec_h264 ! videoconvert ! "
        f"videoscale ! video/x-raw,format=BGR,width=640,height=480,framerate=15/1 ! "
        f"appsink sync=false max-buffers=1 drop=true"
    )
    
    pipelines_to_try = [
        (gst_pipeline, cv2.CAP_GSTREAMER),
        (rtsp_url, cv2.CAP_FFMPEG),
        (rtsp_url, cv2.CAP_ANY)
    ]
    
    for i, (pipeline, backend) in enumerate(pipelines_to_try):
        print(f"  Trying method {i+1}...")
        try:
            cap = cv2.VideoCapture(pipeline, backend)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✓ RTSP working with method {i+1}")
                    return cap  # <-- Return the VideoCapture object
                else:
                    cap.release()
        except Exception as e:
            print(f"  Method {i+1} error: {e}")
            if cap:
                cap.release()
    
    print("❌ Failed to open RTSP stream.")
    return None