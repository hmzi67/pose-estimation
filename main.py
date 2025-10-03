import cv2
import numpy as np
import time
import sys
from headpose_extractor import HeadPoseExtractor
from rep_detector import RepDetector, FlexionExtensionDetector, LateralFlexionDetector
from feedback_engine import FeedbackEngine

# Global animation state
animation_state = {
    'active': False,
    'start_time': 0,
    'rep_count': 0,
    'duration': 1.2
}

def start_rep_animation(rep_count):
    """Start the rep completion animation (non-blocking)"""
    global animation_state
    animation_state['active'] = True
    animation_state['start_time'] = time.time()
    animation_state['rep_count'] = rep_count

def draw_rep_animation_overlay(frame):
    """Draw animation overlay on frame if animation is active"""
    global animation_state
    
    if not animation_state['active']:
        return frame
    
    current_time = time.time()
    elapsed = current_time - animation_state['start_time']
    progress = elapsed / animation_state['duration']
    
    # Stop animation if duration exceeded
    if progress >= 1.0:
        animation_state['active'] = False
        return frame
    
    # Create copy for animation
    anim_frame = frame.copy()
    display_height, display_width = frame.shape[:2]
    center_x = display_width // 2
    center_y = display_height // 2
    
    # Animation phases
    if progress < 0.3:  # Growing phase
        scale = progress / 0.3
    elif progress < 0.7:  # Stable phase
        scale = 1.0
    else:  # Shrinking phase
        scale = 1.0 - ((progress - 0.7) / 0.3) * 0.2
    
    # Circle properties
    radius = int(80 * scale)
    circle_color = (0, 255, 0)  # Green
    text_color = (255, 255, 255)  # White
    
    # Draw circle with border
    cv2.circle(anim_frame, (center_x, center_y), radius + 5, (0, 0, 0), -1)  # Black border
    cv2.circle(anim_frame, (center_x, center_y), radius, circle_color, -1)    # Green fill
    
    # Add rep number
    rep_text = str(animation_state['rep_count'])
    font_scale = 3.0 * scale
    thickness = int(6 * scale)
    
    if thickness > 0:  # Avoid zero thickness
        # Get text size for centering
        text_size = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        
        # Draw text
        cv2.putText(anim_frame, rep_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    # Add "REP" label above circle
    if progress > 0.2 and radius > 20:
        rep_label = "REP"
        label_scale = 1.0 * scale
        label_size = cv2.getTextSize(rep_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 2)[0]
        label_x = center_x - label_size[0] // 2
        label_y = center_y - radius - 20
        
        cv2.putText(anim_frame, rep_label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 2)
    
    return anim_frame

def create_dual_display(live_frame, ref_frame, live_angles, rep_count, ref_frame_count, ref_total_frames):
    """Create a side-by-side display of live feed and reference video"""
    
    # Target display size for full screen (adjust based on common screen resolutions)
    display_height = 1080
    display_width = 1920
    
    # Each video will take half the width
    video_width = display_width // 2
    video_height = display_height
    
    # Resize live frame
    live_resized = cv2.resize(live_frame, (video_width, video_height))
    
    # Create or resize reference frame
    if ref_frame is not None:
        ref_resized = cv2.resize(ref_frame, (video_width, video_height))
    else:
        # Create a placeholder if no reference video
        ref_resized = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        cv2.putText(ref_resized, "No Reference Video", (video_width//4, video_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Combine frames horizontally
    combined_frame = np.hstack([live_resized, ref_resized])
    
    # Add labels and information
    # Live feed label
    cv2.putText(combined_frame, "LIVE FEED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(combined_frame, f"Reps: {rep_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Reference video label
    cv2.putText(combined_frame, "REFERENCE", (video_width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    if ref_total_frames > 0:
        progress = (ref_frame_count / ref_total_frames) * 100
        cv2.putText(combined_frame, f"Progress: {progress:.1f}%", (video_width + 20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add live angles if available (positioned to not interfere with feedback overlay)
    if live_angles is not None:
        y_start = video_height - 120
        cv2.putText(combined_frame, f"Yaw: {live_angles[0]:.1f}°", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"Pitch: {live_angles[1]:.1f}°", (20, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"Roll: {live_angles[2]:.1f}°", (20, y_start + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(combined_frame, "No face detected", (20, video_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add controls information
    controls_y = video_height - 30
    cv2.putText(combined_frame, "Controls: Q=Quit | F=Fullscreen | A=Audio | R=Restart Ref", 
               (20, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return combined_frame

def create_flexion_extension_dual_display(live_frame, ref_frame, live_angles, rep_count, ref_frame_count, ref_total_frames):
    """Create a side-by-side display of live feed and reference video for flexion-extension exercise"""
    
    # Target display size for full screen (adjust based on common screen resolutions)
    display_height = 1080
    display_width = 1920
    
    # Each video will take half the width
    video_width = display_width // 2
    video_height = display_height
    
    # Resize live frame
    live_resized = cv2.resize(live_frame, (video_width, video_height))
    
    # Create or resize reference frame
    if ref_frame is not None:
        ref_resized = cv2.resize(ref_frame, (video_width, video_height))
    else:
        # Create a placeholder if no reference video
        ref_resized = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        cv2.putText(ref_resized, "No Reference Video", (video_width//4, video_height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Combine frames horizontally
    combined_frame = np.hstack([live_resized, ref_resized])
    
    # Add labels and information
    # Live feed label
    cv2.putText(combined_frame, "LIVE FEED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(combined_frame, f"Reps: {rep_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Reference video label
    cv2.putText(combined_frame, "REFERENCE (UP-DOWN)", (video_width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    if ref_total_frames > 0:
        progress = (ref_frame_count / ref_total_frames) * 100
        cv2.putText(combined_frame, f"Progress: {progress:.1f}%", (video_width + 20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add live angles if available (positioned to not interfere with feedback overlay)
    if live_angles is not None:
        y_start = video_height - 120
        cv2.putText(combined_frame, f"Yaw: {live_angles[0]:.1f}°", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"Pitch: {live_angles[1]:.1f}°", (20, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_frame, f"Roll: {live_angles[2]:.1f}°", (20, y_start + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(combined_frame, "No face detected", (20, video_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add controls information
    controls_y = video_height - 30
    cv2.putText(combined_frame, "Controls: Q=Quit | F=Fullscreen | A=Audio | R=Restart Ref", 
               (20, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Add exercise instructions
    cv2.putText(combined_frame, "Exercise: CENTER → UP → CENTER → DOWN → CENTER", 
               (video_width + 20, video_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return combined_frame

def show_exercise_menu():
    """Display exercise selection menu and return user choice"""
    print("\n" + "="*50)
    print("    CERVICAL EXERCISE MONITORING SYSTEM")
    print("="*50)
    print("\nAvailable Exercises:")
    print("1. Rotation Exercise (Left-Right)")
    print("2. Flexion-Extension (Up-Down)")
    print("3. Lateral Flexion (Side Bending)")
    print("4. Combined Movements [Coming Soon]")
    print("5. Exit")
    print("\n" + "-"*50)
    
    while True:
        try:
            choice = input("Select an exercise (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            else:
                print("Please enter a valid option (1-5)")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            return 5

def rotation_exercise(checkpoint_path="model/state_of_the_art.pth", sample_video_path="rotation/ref_2.mp4"):

    headpose = HeadPoseExtractor(checkpoint_path)
    # Instantiate RepDetector with explicit pitch/roll safety window matching requirements
    rep_detector = RepDetector(pitch_min=-5.0, pitch_max=10.0, roll_min=-5.0, roll_max=10.0)
    # Initialize feedback engine with the same thresholds as the rep detector
    feedback = FeedbackEngine(
        tolerance_base=4, 
        angle_tolerance=10, 
        yaw_threshold=rep_detector.yaw_threshold,
        center_threshold=rep_detector.center_threshold
    )

    print("✅ System initialized - Live feedback mode with sample demonstration video.")

    baseline_angles = calibrate_user(headpose)
    print(f"✅ Calibration complete. Baseline: Yaw={baseline_angles[0]:.1f}°, Pitch={baseline_angles[1]:.1f}°, Roll={baseline_angles[2]:.1f}°")
    
    # Small delay to ensure window system is ready
    time.sleep(0.5)

    # Initialize live camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    # reduce camera buffer and try to limit FPS to reduce input lag
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    
    # Reference video removed

    frame_buffer = []
    rep_count = 0
    print("Controls:")
    print("  'q' - quit")
    print("  'f' - toggle fullscreen")
    print("  'a' - toggle audio feedback")
    print("Ready to start live feedback. A sample demo will loop on the right.")
    
    # Create a fullscreen window
    window_name = "Cervical Rotation Exercise"
    print(f"Creating window: {window_name}")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get screen dimensions and set fullscreen by default
    screen_width = 1920  # Default, will be adjusted
    screen_height = 1080  # Default, will be adjusted
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True
        print("Window set to fullscreen")
    except Exception as e:
        print(f"Fullscreen failed: {e}")
        cv2.resizeWindow(window_name, screen_width, screen_height)
        is_fullscreen = False

    # Open sample video (display only, no processing) with time-synchronized playback
    sample_cap = None
    sample_fps = 0
    sample_frame = None
    sample_accum = 0.0
    sample_last_time = time.time()
    if sample_video_path:
        sc = cv2.VideoCapture(sample_video_path)
        if sc.isOpened():
            sample_cap = sc
            # Fetch FPS; guard against zero
            sample_fps = sc.get(cv2.CAP_PROP_FPS) or 30.0
            if sample_fps <= 1e-2:
                sample_fps = 30.0
            print(f"🎬 Sample video loaded: {sample_video_path} ({sample_fps:.2f} FPS)")
        else:
            print(f"⚠️ Could not open sample video: {sample_video_path}")
    else:
        print("ℹ️ No sample video path provided.")

    while True:
        loop_start = time.time()
        
        # Read live frame
        ret, live_frame = cap.read()
        if not ret:
            break
        live_frame = cv2.flip(live_frame, 1)
        
    # Single feed only

        # prepare a smaller frame for headpose processing to speed up inference
        try:
            proc_scale = 0.5
            ph, pw = int(live_frame.shape[0] * proc_scale), int(live_frame.shape[1] * proc_scale)
            proc_frame = cv2.resize(live_frame, (pw, ph))
        except Exception:
            proc_frame = live_frame

        live_angles = headpose.process_frame(proc_frame)

        # Prepare live pane (left)
        target_h = 1080
        half_w = 960
        live_resized = cv2.resize(live_frame, (half_w, target_h))
        cv2.putText(live_resized, "LIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)
        cv2.putText(live_resized, f"Reps: {rep_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if live_angles is not None:
            y_base = target_h - 120
            cv2.putText(live_resized, f"Yaw: {live_angles[0]:.1f}°", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(live_resized, f"Pitch: {live_angles[1]:.1f}°", (20, y_base+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(live_resized, f"Roll: {live_angles[2]:.1f}°", (20, y_base+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(live_resized, "No face detected", (20, target_h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Sample pane (right) - advance frames based on elapsed real time to achieve native speed
        if sample_cap:
            now = time.time()
            dt = now - sample_last_time
            sample_last_time = now
            sample_accum += dt
            frame_period = 1.0 / sample_fps if sample_fps > 0 else 1/30.0
            advanced = False
            while sample_accum >= frame_period:
                ret_s, sf = sample_cap.read()
                if not ret_s:
                    sample_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_s, sf = sample_cap.read()
                if ret_s:
                    sample_frame = sf
                sample_accum -= frame_period
                advanced = True
            # If no frame yet (first loop) attempt immediate read
            if sample_frame is None:
                ret_init, sf = sample_cap.read()
                if ret_init:
                    sample_frame = sf
            if sample_frame is not None:
                sample_resized = cv2.resize(sample_frame, (half_w, target_h))
            else:
                sample_resized = np.zeros((target_h, half_w, 3), dtype=np.uint8)
        else:
            sample_resized = np.zeros((target_h, half_w, 3), dtype=np.uint8)
            cv2.putText(sample_resized, "NO SAMPLE", (200, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,200,200), 3)

        # Overlay labels/instructions on sample pane
        cv2.putText(sample_resized, "SAMPLE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 4)
        cv2.putText(sample_resized, "Follow the demo: LEFT → CENTER → RIGHT → CENTER", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(sample_resized, "Keep chin level. Turn only the head.", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        display_frame = np.hstack([live_resized, sample_resized])
        cv2.putText(display_frame, "Controls: Q=Quit | F=Fullscreen | A=Audio | R=Restart Demo", (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        if live_angles is not None:
            # Subtract baseline for relative movement
            live_angles = live_angles - baseline_angles
            frame_buffer.append({
                "t": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 
                "yaw": live_angles[0], 
                "pitch": live_angles[1], 
                "roll": live_angles[2]
            })
            if rep_detector.detect_rep(frame_buffer):
                rep_count += 1
                print(f"✅ Repetition {rep_count} completed!")
                start_rep_animation(rep_count)
            metrics = {"status": "live_tracking"}
            feedback.display_feedback(display_frame, metrics, live_angles, None, rep_count=rep_count, rep_state=rep_detector.state)
            feedback.display_summary(rep_count, metrics)
        display_frame = draw_rep_animation_overlay(display_frame)
        cv2.imshow(window_name, display_frame)
        
        # Cap the loop to a target FPS to avoid bursting work and reduce CPU usage
        try:
            cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        except Exception:
            cam_fps = 30.0
        target_fps = max(15, cam_fps)
        frame_interval = 1.0 / float(target_fps)
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < frame_interval:
            time.sleep(frame_interval - loop_elapsed)

        key = cv2.waitKey(1) & 0xFF
        # Handle key presses
        if key == ord('q'):
            return
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            except Exception:
                pass
        elif key == ord('a'):
            audio_enabled = feedback.toggle_audio_feedback()
            print(f"Audio feedback {'enabled' if audio_enabled else 'disabled'}")
        elif key == ord('r'):
            if sample_cap:
                sample_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                sample_accum = 0.0
                sample_frame = None
                sample_last_time = time.time()
                print("🔄 Sample demo restarted")

    cap.release()
    if sample_cap:
        sample_cap.release()
    cv2.destroyAllWindows()
    headpose.close()

def calibrate_user(headpose, duration=3):
    cap = cv2.VideoCapture(0)
    angles_list = []
    start_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    
    calibration_window = "Calibration - Look straight ahead"
    cv2.namedWindow(calibration_window, cv2.WINDOW_NORMAL)

    print(f"Starting {duration} second calibration. Please look straight ahead and stay still...")

    while (cv2.getTickCount() - start_time) / freq < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip the frame for mirror effect during calibration
        frame = cv2.flip(frame, 1)
        
        # Add countdown text
        elapsed = (cv2.getTickCount() - start_time) / freq
        remaining = duration - elapsed
        
        # Add text overlay
        cv2.putText(frame, f"Calibration: {remaining:.1f}s remaining", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Look straight ahead and stay still", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        angles = headpose.process_frame(frame)
        if angles is not None:
            angles_list.append(angles)
            # Show current angles
            cv2.putText(frame, f"Yaw: {angles[0]:.1f}°", (10, frame.shape[0] - 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Pitch: {angles[1]:.1f}°", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Roll: {angles[2]:.1f}°", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(calibration_window, frame)
        cv2.waitKey(1)

    cap.release()
    # Only destroy the calibration window, not all windows
    cv2.destroyWindow(calibration_window)
    return np.mean(angles_list, axis=0) if angles_list else np.zeros(3)

def flexion_extension_exercise(checkpoint_path="model/state_of_the_art.pth", sample_video_path="rotation/up-and-down-converted.mp4"):

    """Flexion-Extension (up-down) exercise with yaw/roll constraints"""
    headpose = HeadPoseExtractor(checkpoint_path)
    # Instantiate FlexionExtensionDetector with yaw/roll safety window
    rep_detector = FlexionExtensionDetector(pitch_threshold=15.0, center_threshold=5.0, 
                                           yaw_min=-5.0, yaw_max=5.0, roll_min=-5.0, roll_max=5.0)
    # Initialize feedback engine with flexion-extension specific settings
    feedback = FeedbackEngine(
        tolerance_base=4, 
        angle_tolerance=10, 
        yaw_threshold=5.0,  # Stricter yaw constraint for flexion-extension
        center_threshold=5.0
    )

    print("✅ Flexion-Extension Exercise initialized - Look up and down movements (demo video enabled).")

    baseline_angles = calibrate_user(headpose)
    print(f"✅ Calibration complete. Baseline: Yaw={baseline_angles[0]:.1f}°, Pitch={baseline_angles[1]:.1f}°, Roll={baseline_angles[2]:.1f}°")
    
    # Small delay to ensure window system is ready
    time.sleep(0.5)

    # Initialize live camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    # reduce camera buffer and try to limit FPS to reduce input lag
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    
    # Reference video removed

    frame_buffer = []
    rep_count = 0
    print("Controls:")
    print("  'q' - quit")
    print("  'f' - toggle fullscreen")
    print("  'a' - toggle audio feedback")
    print("Ready to start Flexion-Extension exercise. A sample demo will loop on the right.")
    print("Exercise: CENTER → UP → CENTER → DOWN → CENTER to complete one rep")
    print("Keep your head straight (no turning left/right or tilting)")
    
    # Create a fullscreen window
    window_name = "Flexion-Extension Exercise"
    print(f"Creating window: {window_name}")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Get screen dimensions and set fullscreen by default
    screen_width = 1920  # Default, will be adjusted
    screen_height = 1080  # Default, will be adjusted
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True
        print("Window set to fullscreen")
    except Exception as e:
        print(f"Fullscreen failed: {e}")
        cv2.resizeWindow(window_name, screen_width, screen_height)
        is_fullscreen = False

    # Open sample demo video (display only) with native FPS timing
    sample_cap = None
    sample_fps = 0
    sample_frame = None
    sample_accum = 0.0
    sample_last_time = time.time()
    if sample_video_path:
        sc = cv2.VideoCapture(sample_video_path)
        if sc.isOpened():
            sample_cap = sc
            sample_fps = sc.get(cv2.CAP_PROP_FPS) or 30.0
            if sample_fps <= 1e-2:
                sample_fps = 30.0
            print(f"🎬 Flex/Ext sample video loaded: {sample_video_path} ({sample_fps:.2f} FPS)")
        else:
            print(f"⚠️ Could not open flexion-extension sample video: {sample_video_path}")
    else:
        print("ℹ️ No flexion-extension sample video path provided.")

    while True:
        loop_start = time.time()
        
        # Read live frame
        ret, live_frame = cap.read()
        if not ret:
            break
        live_frame = cv2.flip(live_frame, 1)

        # prepare a smaller frame for headpose processing to speed up inference
        try:
            proc_scale = 0.5
            ph, pw = int(live_frame.shape[0] * proc_scale), int(live_frame.shape[1] * proc_scale)
            proc_frame = cv2.resize(live_frame, (pw, ph))
        except Exception:
            proc_frame = live_frame
        live_angles = headpose.process_frame(proc_frame)

        # Layout parameters
        target_h = 1080
        half_w = 960
        live_resized = cv2.resize(live_frame, (half_w, target_h))
        cv2.putText(live_resized, "LIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,255), 3)
        cv2.putText(live_resized, f"Reps: {rep_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if live_angles is not None:
            y_base = target_h - 120
            cv2.putText(live_resized, f"Yaw: {live_angles[0]:.1f}°", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(live_resized, f"Pitch: {live_angles[1]:.1f}°", (20, y_base+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(live_resized, f"Roll: {live_angles[2]:.1f}°", (20, y_base+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(live_resized, "No face detected", (20, target_h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Sample (demo) pane right - time-synced playback
        if sample_cap:
            now = time.time()
            dt = now - sample_last_time
            sample_last_time = now
            sample_accum += dt
            frame_period = 1.0 / sample_fps if sample_fps > 0 else 1/30.0
            while sample_accum >= frame_period:
                ret_s, sf = sample_cap.read()
                if not ret_s:
                    sample_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_s, sf = sample_cap.read()
                if ret_s:
                    sample_frame = sf
                sample_accum -= frame_period
            if sample_frame is None:
                ret_init, sf = sample_cap.read()
                if ret_init:
                    sample_frame = sf
            if sample_frame is not None:
                sample_resized = cv2.resize(sample_frame, (half_w, target_h))
            else:
                sample_resized = np.zeros((target_h, half_w, 3), dtype=np.uint8)
        else:
            sample_resized = np.zeros((target_h, half_w, 3), dtype=np.uint8)
            cv2.putText(sample_resized, "NO SAMPLE", (200, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,200,200), 3)

        # Demo guidance overlays
        cv2.putText(sample_resized, "SAMPLE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,255), 3)
        cv2.putText(sample_resized, "Sequence: CENTER → UP → CENTER → DOWN → CENTER", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(sample_resized, "Keep head straight (no tilt/turn)", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(sample_resized, "Smooth controlled motion", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

        display_frame = np.hstack([live_resized, sample_resized])
        cv2.putText(display_frame, "Controls: Q=Quit | F=Fullscreen | A=Audio | R=Restart Demo", (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        if live_angles is not None:
            live_angles = live_angles - baseline_angles
            frame_buffer.append({
                "t": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 
                "yaw": live_angles[0], 
                "pitch": live_angles[1], 
                "roll": live_angles[2]
            })
            if rep_detector.detect_rep(frame_buffer):
                rep_count += 1
                print(f"✅ Flexion-Extension Repetition {rep_count} completed!")
                start_rep_animation(rep_count)
            metrics = {"status": "flexion_extension_tracking"}
            feedback.display_flexion_extension_feedback(display_frame, metrics, live_angles, rep_count=rep_count, rep_state=rep_detector.state)
            feedback.display_summary(rep_count, metrics)
        display_frame = draw_rep_animation_overlay(display_frame)
        cv2.imshow(window_name, display_frame)
        
        # Cap the loop to a target FPS to avoid bursting work and reduce CPU usage
        try:
            cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        except Exception:
            cam_fps = 30.0
        target_fps = max(15, cam_fps)
        frame_interval = 1.0 / float(target_fps)
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < frame_interval:
            time.sleep(frame_interval - loop_elapsed)

        key = cv2.waitKey(1) & 0xFF
        # Handle key presses
        if key == ord('q'):
            return
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            except Exception:
                pass
        elif key == ord('a'):
            audio_enabled = feedback.toggle_audio_feedback()
            print(f"Audio feedback {'enabled' if audio_enabled else 'disabled'}")
        elif key == ord('r'):
            if sample_cap:
                sample_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                sample_accum = 0.0
                sample_frame = None
                sample_last_time = time.time()
                print("🔄 Flex/Ext sample demo restarted")

    cap.release()
    if sample_cap:
        sample_cap.release()
    cv2.destroyAllWindows()
    headpose.close()

def create_flexion_extension_display(live_frame, rep_count, rep_detector):
    """Create a display for flexion-extension exercise"""
    
    # Target display size for full screen
    display_height = 1080
    display_width = 1920
    
    # Resize live frame to full display
    display_frame = cv2.resize(live_frame, (display_width, display_height))
    
    # Add exercise title
    cv2.putText(display_frame, "FLEXION-EXTENSION EXERCISE", (display_width//2 - 300, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    # Add rep counter
    cv2.putText(display_frame, f"Reps: {rep_count}", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Add current state information
    state_info = rep_detector.get_state_info()
    cv2.putText(display_frame, f"State: {state_info['state_name']}", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    # Add exercise instructions
    instructions = [
        "Exercise Instructions:",
        "1. Look DOWN (chin to chest)",
        "2. Return to CENTER",
        "3. Look UP (face to ceiling)", 
        "4. Return to CENTER",
        "",
        "Keep head straight - no turning or tilting!"
    ]
    
    for i, instruction in enumerate(instructions):
        color = (255, 255, 255) if instruction else (0, 0, 0)
        if instruction:
            cv2.putText(display_frame, instruction, (50, 250 + i * 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add progress indicator
    progress = state_info['progress'] * 100
    cv2.putText(display_frame, f"Progress: {progress:.1f}%", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    # Add controls information
    controls_y = display_height - 30
    cv2.putText(display_frame, "Controls: Q=Quit | F=Fullscreen | A=Audio", 
               (50, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    return display_frame

def lateral_flexion_exercise(checkpoint_path="model/state_of_the_art.pth", sample_video_path="rotation/tilled.mp4"):
    """Lateral Flexion (side bending) exercise with demo sample video.

    Sequence: CENTER → LEFT TILT → CENTER → RIGHT TILT → CENTER
    Detection axis: roll (negative = left tilt, positive = right tilt after mirroring).
    Neutral constraints enforced: yaw ∈ [-15°, +15°], pitch ∈ [-15°, +15°].
    A demonstration video (tilled.mp4) is shown on the right to guide the user, similar
    to rotation and flexion-extension exercises. The sample video is display-only and
    not used for processing.
    """
    headpose = HeadPoseExtractor(checkpoint_path)
    # Detector with requested yaw/pitch neutrality window
    rep_detector = LateralFlexionDetector(roll_threshold=15.0, center_threshold=5.0,
                                          yaw_min=-15.0, yaw_max=15.0, pitch_min=-15.0, pitch_max=15.0)
    feedback = FeedbackEngine(
        tolerance_base=4,
        angle_tolerance=10,
        yaw_threshold=15.0,
        center_threshold=5.0
    )

    print("✅ Lateral Flexion Exercise initialized (yaw/pitch neutral window -15° to +15°).")
    baseline_angles = calibrate_user(headpose)
    print(f"✅ Calibration complete. Baseline: Yaw={baseline_angles[0]:.1f}°, Pitch={baseline_angles[1]:.1f}°, Roll={baseline_angles[2]:.1f}°")
    time.sleep(0.4)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    frame_buffer = []
    rep_count = 0
    window_name = "Lateral Flexion Exercise"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = True
    except Exception:
        is_fullscreen = False
    print("Controls: q quit | f fullscreen | a audio | r restart demo")
    print("Sequence: CENTER → LEFT TILT → CENTER → RIGHT TILT → CENTER")
    print("Maintain neutral yaw/pitch within ±15°.")

    # Sample demo video (display only) timing management
    sample_cap = None
    sample_fps = 0
    sample_frame = None
    sample_accum = 0.0
    sample_last_time = time.time()
    if sample_video_path:
        sc = cv2.VideoCapture(sample_video_path)
        if sc.isOpened():
            sample_cap = sc
            sample_fps = sc.get(cv2.CAP_PROP_FPS) or 30.0
            if sample_fps <= 1e-2:
                sample_fps = 30.0
            print(f"🎬 Lateral flexion sample video loaded: {sample_video_path} ({sample_fps:.2f} FPS)")
        else:
            print(f"⚠️ Could not open lateral flexion sample video: {sample_video_path}")
    else:
        print("ℹ️ No lateral flexion sample video path provided.")

    while True:
        loop_start = time.time()
        ret, live_frame = cap.read()
        if not ret:
            break
        live_frame = cv2.flip(live_frame, 1)

        # Downscale for processing
        try:
            proc_scale = 0.5
            ph, pw = int(live_frame.shape[0]*proc_scale), int(live_frame.shape[1]*proc_scale)
            proc_frame = cv2.resize(live_frame, (pw, ph))
        except Exception:
            proc_frame = live_frame
        live_angles = headpose.process_frame(proc_frame)

        # Layout (side-by-side like other exercises)
        target_h = 1080
        half_w = 960
        live_resized = cv2.resize(live_frame, (half_w, target_h))
        cv2.putText(live_resized, "LIVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,255), 3)
        cv2.putText(live_resized, f"Reps: {rep_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        rel_angles = None
        if live_angles is not None:
            rel_angles = live_angles - baseline_angles
            yaw_ok = -15.0 < rel_angles[0] < 15.0
            pitch_ok = -15.0 < rel_angles[1] < 15.0
            color_yaw = (0,255,0) if yaw_ok else (0,0,255)
            color_pitch = (0,255,0) if pitch_ok else (0,0,255)
            y_base = target_h - 120
            cv2.putText(live_resized, f"Yaw: {rel_angles[0]:.1f}°", (20, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_yaw, 2)
            cv2.putText(live_resized, f"Pitch: {rel_angles[1]:.1f}°", (20, y_base+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_pitch, 2)
            cv2.putText(live_resized, f"Roll: {rel_angles[2]:.1f}°", (20, y_base+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(live_resized, "No face detected", (20, target_h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Sample pane (right) - time-synchronized playback
        if sample_cap:
            now = time.time()
            dt = now - sample_last_time
            sample_last_time = now
            sample_accum += dt
            frame_period = 1.0 / sample_fps if sample_fps > 0 else 1/30.0
            while sample_accum >= frame_period:
                ret_s, sf = sample_cap.read()
                if not ret_s:
                    sample_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_s, sf = sample_cap.read()
                if ret_s:
                    sample_frame = sf
                sample_accum -= frame_period
            if sample_frame is None:
                ret_init, sf = sample_cap.read()
                if ret_init:
                    sample_frame = sf
            if sample_frame is not None:
                sample_resized = cv2.resize(sample_frame, (half_w, target_h))
            else:
                sample_resized = np.zeros((target_h, half_w, 3), dtype=np.uint8)
        else:
            sample_resized = np.zeros((target_h, half_w, 3), dtype=np.uint8)
            cv2.putText(sample_resized, "NO SAMPLE", (200, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200,200,200), 3)

        # Guidance overlays on sample pane
        cv2.putText(sample_resized, "SAMPLE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,255), 3)
        cv2.putText(sample_resized, "Sequence: CENTER → LEFT → CENTER → RIGHT → CENTER", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(sample_resized, "Keep face forward (avoid yaw)", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(sample_resized, "No nodding (limit pitch)", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(sample_resized, "Smooth controlled tilt", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

        display_frame = np.hstack([live_resized, sample_resized])
        cv2.putText(display_frame, "Controls: Q=Quit | F=Fullscreen | A=Audio | R=Restart Demo", (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        if rel_angles is not None:
            frame_buffer.append({
                "t": cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0,
                "yaw": rel_angles[0],
                "pitch": rel_angles[1],
                "roll": rel_angles[2]
            })
            if rep_detector.detect_rep(frame_buffer):
                rep_count += 1
                print(f"✅ Lateral Flexion Repetition {rep_count} completed!")
                start_rep_animation(rep_count)
            metrics = {"status": "lateral_flexion_tracking"}
            feedback.display_feedback(display_frame, metrics, rel_angles, None, rep_count=rep_count, rep_state=rep_detector.state)
            feedback.display_summary(rep_count, metrics)

        display_frame = draw_rep_animation_overlay(display_frame)
        cv2.imshow(window_name, display_frame)

        # Frame pacing
        try:
            cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        except Exception:
            cam_fps = 30.0
        target_fps = max(15, cam_fps)
        frame_interval = 1.0/float(target_fps)
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < frame_interval:
            time.sleep(frame_interval - loop_elapsed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            is_fullscreen = not is_fullscreen
            try:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            except Exception:
                pass
        elif key == ord('a'):
            audio_enabled = feedback.toggle_audio_feedback()
            print(f"Audio feedback {'enabled' if audio_enabled else 'disabled'}")
        elif key == ord('r') and sample_cap:
            sample_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            sample_accum = 0.0
            sample_frame = None
            sample_last_time = time.time()
            print("🔄 Lateral flexion sample demo restarted")

    cap.release()
    if sample_cap:
        sample_cap.release()
    cv2.destroyWindow(window_name)
    headpose.close()

def combined_movements_exercise():
    """Placeholder for combined movements exercise"""
    print("\n🚧 Combined Movements Exercise")
    print("This exercise will combine multiple neck movement patterns.")
    print("Coming soon! Returning to main menu...")
    time.sleep(2)

def main():
    """Main application entry point with exercise selection menu"""
    while True:
        choice = show_exercise_menu()
        
        if choice == 1:
            print("\n✅ Starting Rotation Exercise...")
            try:
                rotation_exercise()
            except KeyboardInterrupt:
                print("\n\nRotation exercise interrupted by user.")
            except Exception as e:
                print(f"\n❌ Error in rotation exercise: {e}")
            
        elif choice == 2:
            print("\n✅ Starting Flexion-Extension Exercise...")
            try:
                flexion_extension_exercise()
            except KeyboardInterrupt:
                print("\n\nFlexion-Extension exercise interrupted by user.")
            except Exception as e:
                print(f"\n❌ Error in Flexion-Extension exercise: {e}")
            
        elif choice == 3:
            lateral_flexion_exercise()
            
        elif choice == 4:
            combined_movements_exercise()
            
        elif choice == 5:
            print("\n👋 Thank you for using the Cervical Exercise Monitoring System!")
            print("Stay healthy and keep exercising! 💪")
            sys.exit(0)
        
        # Ask if user wants to continue
        print("\n" + "-"*50)
        continue_choice = input("Would you like to select another exercise? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n👋 Thank you for using the Cervical Exercise Monitoring System!")
            print("Stay healthy and keep exercising! 💪")
            break

if __name__ == "__main__":
    main()