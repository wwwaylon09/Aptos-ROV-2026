import os

HOST = ""
PORT = 5000
SOCKET_TIMEOUT_SECONDS = float(os.getenv("ROV_SOCKET_TIMEOUT_SECONDS", "1.0"))
MAX_FRAME_SIZE = int(os.getenv("ROV_MAX_FRAME_SIZE", "65536"))
NO_DATA_FAILSAFE_SECONDS = float(os.getenv("ROV_NO_DATA_FAILSAFE_SECONDS", "0.5"))
MPU_DEADBAND_DEGREES = float(os.getenv("ROV_MPU_DEADBAND_DEGREES", "1.5"))
CONTROL_HEALTH_LOG_INTERVAL_SECONDS = float(os.getenv("ROV_CONTROL_HEALTH_LOG_INTERVAL_SECONDS", "10"))

WEBCAM_HOST = os.getenv("ROV_WEBCAM_HOST", "0.0.0.0")
WEBCAM_PORT = int(os.getenv("ROV_WEBCAM_PORT", "5001"))
HUD_UPDATE_INTERVAL_SECONDS = float(os.getenv("ROV_HUD_UPDATE_INTERVAL_SECONDS", "0.2"))
RECONNECT_INTERVAL = float(os.getenv("ROV_CAMERA_RECONNECT_INTERVAL_SECONDS", "3.0"))
VIDEO_HEALTH_LOG_INTERVAL_SECONDS = float(os.getenv("ROV_VIDEO_HEALTH_LOG_INTERVAL_SECONDS", "10"))

CAMERA_CONFIG = {
    "Camera 1": {
        "id": "camera-1",
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1:1.0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 55,
    },
    "Camera 2": {
        "id": "camera-2",
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 55,
    },
    "Camera 3": {
        "id": "camera-3",
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0",
        "width": 640,
        "height": 480,
        "fps": 30,
        "jpeg_quality": 55,
    },
    "Camera 4": {
        "id": "camera-4",
        "device_path": "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:2:1.0-video-index0",
        "width": 1920,
        "height": 1080,
        "fps": 4,
        "jpeg_quality": 50,
    },
}
