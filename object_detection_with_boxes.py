import pyzed.sl as sl
import cv2
import numpy as np
import argparse
import onnxruntime as ort
import ast
import os
from scipy.spatial.transform import Rotation as R

# Color palette for different object labels
LABEL_COLORS = {}
    
def get_label_color(label):
    """Return a consistent color per label string."""
    if label not in LABEL_COLORS:
        rng = np.random.default_rng(abs(hash(label)) % (2**32))
        LABEL_COLORS[label] = tuple(int(c) for c in rng.integers(100, 255, size=3))
    return LABEL_COLORS[label]

def draw_bounding_box_2d(img, bbox_2d, label, confidence, color):
    """Draw a 2D bounding box with label and confidence onto img."""
    pts = np.array(bbox_2d, dtype=np.int32).reshape((-1, 1, 2))

    # Draw filled polygon outline
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    # Semi-transparent fill
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Label background + text
    x, y = int(bbox_2d[0][0]), int(bbox_2d[0][1]) - 10
    text = f"{label} {confidence}%"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline), color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

def main(IP_ADDRESS, PORT, visualize=False, resize_frame=False, model_path=None, show_validation=False, rotation=90, save_training=False, output_dir="training_data", local_camera=False):
    #swap x and for buoy positions to match world
    buoy_locations = {'red_buoy': np.array([6, -14, 0]),
                      'blue_buoy': np.array([0, -13.05542, 0]),
                      'green_buoy': np.array([-6, -14, 0])
                      }

    zed = sl.Camera()

    init_params = sl.InitParameters()
    if not local_camera:
        init_params.set_from_stream(IP_ADDRESS, PORT)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.sdk_verbose = 1
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    #init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    print(f"Initial coordinates {init_params.coordinate_system}")

    err = zed.open(init_params)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open: {repr(err)}. Exiting.")
        exit()

    obj_param = sl.ObjectDetectionParameters()
    

    #print(f"camera_pose: {camera_pose}")
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = True

    #set camera position information
    initial_translation = sl.Translation()
    #cam_location = (0, -10.45016, 0.39712)
    initial_translation.init_vector(0, -10.45016, 0.39712)
    initial_transform = sl.Transform()
    initial_transform.set_translation(initial_translation)
    #cam_location = (0, 0, 0)
    #print(f"model path is {model_path}")
    initial_rotation = sl.Rotation()

    r = R.from_euler('xyz', [0, 0, rotation], degrees=True)  # set your known roll, pitch, yaw
    R_align = R.from_euler('z', rotation, degrees=True)
    rot_matrix = r.as_matrix()
    m = initial_transform.m  # 4x4 matrix
    m[:3, :3] = rot_matrix
    m[0, 3] = 0
    m[1, 3] = -10.45016
    m[2, 3] = 0.39712
    print(dir(sl.PositionalTrackingParameters()))

    initial_transform.init_matrix(initial_transform)

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.set_initial_world_transform(initial_transform)
    tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_1
    tracking_params.enable_imu_fusion = False
    if model_path is not None:
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
        obj_param.custom_onnx_file = model_path
        session = ort.InferenceSession(model_path)
        meta = session.get_modelmeta()
        names_raw = meta.custom_metadata_map.get("labels", "{}")
        class_labels = ast.literal_eval(names_raw)
        #obj_param.custom_onnx_input_size = 640
    else:
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

    if obj_param.enable_tracking:
        #zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        tracking_params.set_gravity_as_origin = True
        tracking_params.enable_area_memory = True
        tracking_params.enable_pose_smoothing = False
        err = zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Tracking error: {repr(err)}")
            zed.close()
            exit(1)

    if show_validation:
        camera_pose = sl.Pose()
        zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        initial_cam_location = camera_pose.get_translation(sl.Translation()).get()
        print(f"Initial cam location in world is {initial_cam_location}")
        
    print("Object Detection: Loading Module...")
    err = zed.enable_object_detection(obj_param)
    if err > sl.ERROR_CODE.SUCCESS:
        print(f"Enable object detection: {repr(err)}. Exiting.")
        zed.close()
        exit()

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    #obj_runtime_param.object_detection_output_frame = sl.REFERENCE_FRAME.WORLD
    obj_runtime_param.detection_confidence_threshold = 40
    zed.set_object_detection_runtime_parameters(obj_runtime_param)

    # Mat for retrieving the left camera image
    image_zed = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    #print(dir(sl.REFERENCE_FRAME))
    if save_training:
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        print(f"Saving training data to '{output_dir}/'")

    frame_idx = 0
    print("Press 'q' to quit.")

    while True:
        #print(error1)

        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            continue

        if show_validation:
            camera_pose = sl.Pose()
            zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            translation = camera_pose.get_translation(sl.Translation()).get()
            quat = camera_pose.get_orientation(sl.Orientation()).get()
            camera_rotation = R.from_quat(quat)

            camera_world_pos = np.array(translation)
            yaw_deg = camera_rotation.as_euler('xyz', degrees=True)[2]
            print(f"Tracked position: {translation}")
            print(f"Camera quat: {quat}")
            print(f"Camera yaw from tracking: {yaw_deg:.2f} deg")


        # Retrieve left image as BGRA, convert to BGR for OpenCV
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        if visualize or save_training:
            frame = image_zed.get_data()[:, :, :3].copy()  # drop alpha channel
        

        zed.retrieve_objects(objects)

        if objects.is_new:
            obj_array = objects.object_list
            fps = zed.get_current_fps()
            print(f"{len(obj_array)} Object(s) detected ({fps:.1f} FPS)")

            yolo_labels = []
            clean_frame = frame.copy() if save_training else None
            for obj in obj_array:
                rawid = obj.raw_label
                if model_path is not None:
                    try:
                        label = class_labels.get(rawid, "UNKNOWN")
                    except:
                        label = "UNKNOWN"
                    if show_validation:
                        object_world_pos = buoy_locations[label]
                        expected_relative = object_world_pos - camera_world_pos
                        #expected_in_camera_frame = camera_rotation.inv().apply(expected_relative)
                        #expected_distance = np.linalg.norm(expected_relative)
                        expected_cam_raw = camera_rotation.inv().apply(expected_relative)

                        # Based on your sample output, this is the first remap to test:
                        expected_in_camera_frame = R_align.apply(expected_cam_raw)
                        
                else:
                    label     = repr(obj.label)
                conf      = int(obj.confidence)
                color     = get_label_color(label)
                bbox_2d   = obj.bounding_box_2d
                #rawid = obj.raw_label
                label = f"{rawid}: {label}"

                if save_training and len(bbox_2d) > 0:
                    h, w = frame.shape[:2]
                    pts = np.array(bbox_2d, dtype=np.float32)
                    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
                    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
                    cx = ((x_min + x_max) / 2) / w
                    cy = ((y_min + y_max) / 2) / h
                    bw = (x_max - x_min) / w
                    bh = (y_max - y_min) / h
                    yolo_labels.append(f"{rawid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                # Draw 2D box on frame
                if len(bbox_2d) > 0 and visualize:
                    draw_bounding_box_2d(frame, bbox_2d, label, conf, color)

                # Console output
                pos  = obj.position
                vel  = obj.velocity
                dims = obj.dimensions
                zed_pos = np.array(pos)
                # Convert ZED camera-frame detection -> world frame
                
                if model_path is not None and show_validation:
                    zed_pos_world = camera_rotation.apply(R_align.inv().apply(zed_pos)) + camera_world_pos
                    #zed_pos = np.array(pos)
                    zed_distance = np.linalg.norm(zed_pos)
                    
                    expected_distance = np.linalg.norm(expected_in_camera_frame)
                    error = np.linalg.norm(zed_pos - expected_in_camera_frame)
                    world_error = np.linalg.norm(zed_pos_world - object_world_pos)
                    print(f"Cam world position is ({camera_world_pos[0]:.4f}, {camera_world_pos[1]:.4f}, {camera_world_pos[2]:.4f})")
                    #print(f"Cam expected location in world is ({cam_location[0]:.4f}, {cam_location[1]:.4f}, {cam_location[2]:.4f})")
                    print(f"ZED position: {zed_pos}")
                    #print(f"Expected position: {expected_relative}")
                    print(f"ZED distance: {zed_distance:.2f} m")
                    print(f"Expected camera frame position: {expected_in_camera_frame} m")
                    print(f"Expected distance: {expected_distance:.2f} m")
                    print(f"Position error: {error:.2f} m")
                    print(f"World position error: {world_error:.2f}")
                    print(f"zed_world_pos=({zed_pos_world[0]:.2f}, {zed_pos_world[1]:.2f}, {zed_pos_world[2]:.2f})m ")
                    print(f"expected_world_pos=({object_world_pos[0]:.2f}, {object_world_pos[1]:.2f}, {object_world_pos[2]:.2f})m ")
                
                print(f"  [{label}] conf={conf}% id={int(obj.id)}")
                print(f"    frame_pos=({zed_pos[0]:.2f}, {zed_pos[1]:.2f}, {zed_pos[2]:.2f})m "
                    f"vel=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})m/s "
                    f"dims=({dims[0]:.2f}x{dims[1]:.2f}x{dims[2]:.2f})m")
                

        if save_training and objects.is_new and yolo_labels:
            stem = f"frame_{frame_idx:06d}"
            cv2.imwrite(os.path.join(images_dir, f"{stem}.jpg"), clean_frame)
            with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(yolo_labels))
            frame_idx += 1

        # HUD: FPS counter
        if visualize:
            cv2.putText(frame, f"FPS: {zed.get_current_fps():.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if resize_frame:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            cv2.imshow("ZED Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if visualize:
        cv2.destroyAllWindows()
    zed.disable_object_detection()
    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ZED Object Detector', description='Detects Objects from ZED Cameras using IP Address and port')
    parser.add_argument('--ip_address', '-ip', type=str, default='127.0.0.1', help="Enter IP Address in the form xxx.x.x.x")
    parser.add_argument('--port', type=int, default=30000, help="Enter Port")
    parser.add_argument('--visualize_output', '-v', action='store_true', help='To view output in window')
    parser.add_argument('--scale_down', '-s', action='store_true', help='To resize down to 50 percent of window')
    parser.add_argument('--model_path', '-m', type=str, help="Path to model path for custom detection")
    parser.add_argument('--show_validation', '-sv', action='store_true', help="Show validation for detections")
    parser.add_argument('--rotation', '-r', type=int, default=90, help="choose rotation value")
    parser.add_argument('--save_training_data', '-st', action='store_true', help="Save images and YOLO labels for training")
    parser.add_argument('--output_dir', '-o', type=str, default='training_data', help="Output directory for training images and labels")
    parser.add_argument('--local_camera', '-lc', action='store_true', help="Use a locally connected ZED camera instead of an IP stream")
    args = parser.parse_args()
    main(args.ip_address, args.port, args.visualize_output, args.scale_down, args.model_path, args.show_validation, args.rotation, args.save_training_data, args.output_dir, args.local_camera)
