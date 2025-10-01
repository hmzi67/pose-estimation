import numpy as np


class RepDetector:
    """
    Detects a full round rep defined as: center -> left -> center -> right -> center

    Assumptions:
    - Angles provided in the frame buffer are already relative to the user's baseline (i.e. main.py subtracts baseline).
    - Yaw is used for left/right/center detection.
    - New: pitch and roll constraints can be set so rotations are only counted when
      pitch and roll are within an allowed window. When the constraints are violated
      the detector resets to the waiting state and will not progress or count reps.
    """

    def __init__(self, buffer_size=300, yaw_threshold=15.0, center_threshold=5.0, hold_frames=3,
                 pitch_min=-10.0, pitch_max=10.0, roll_min=-10.0, roll_max=10.0):
        self.buffer_size = buffer_size
        self.yaw_threshold = yaw_threshold
        self.center_threshold = center_threshold
        self.hold_frames = hold_frames

        # Pitch/roll safety window (exclusive lower/upper bounds used in checks)
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.roll_min = roll_min
        self.roll_max = roll_max

        # State machine:
        # 0 = waiting (at/near center)
        # 1 = left detected (yaw <= -yaw_threshold)
        # 2 = returned to center after left
        # 3 = right detected (yaw >= yaw_threshold)
        # When we return to center from state 3, we count a rep and reset to 0
        self.state = 0
        self.buffer = []
        self.state_time = 0  # Time spent in current state for progress tracking
        self.last_state_change = 0  # Time of last state change

    def detect_rep(self, frame_buffer):
        """Update state machine using the most recent frames and return True when a rep completes.

        This function now also enforces pitch/roll constraints: it computes the recent mean
        pitch and roll (over the same number of frames used for yaw smoothing) and if
        either value is outside the configured window, the detector will reset to state 0
        and return False (no rep counted).
        """
        if not frame_buffer:
            return False

        # Keep buffer within size limit
        self.buffer = frame_buffer[-self.buffer_size:]

        # Extract arrays for axes (assume frames contain 'yaw','pitch','roll')
        yaws = np.array([frame["yaw"] for frame in self.buffer])
        pitches = np.array([frame.get("pitch", 0.0) for frame in self.buffer])
        rolls = np.array([frame.get("roll", 0.0) for frame in self.buffer])

        # Use the mean of the last few frames to reduce jitter
        n = min(self.hold_frames, len(yaws))
        recent_mean_yaw = float(np.mean(yaws[-n:]))
        recent_mean_pitch = float(np.mean(pitches[-n:]))
        recent_mean_roll = float(np.mean(rolls[-n:]))

        # Enforce pitch/roll constraints: if violated, reset detector and do not progress
        pitch_ok = (self.pitch_min < recent_mean_pitch < self.pitch_max)
        roll_ok = (self.roll_min < recent_mean_roll < self.roll_max)
        if not (pitch_ok and roll_ok):
            # Reset to waiting center state to avoid partial sequences being counted
            if self.state != 0:
                self.last_state_change = len(self.buffer)
            self.state = 0
            self.state_time = 0
            return False

        rep_detected = False
        state_changed = False

        if self.state == 0:
            # If user is already turned left significantly, allow immediate transition
            if recent_mean_yaw <= -self.yaw_threshold:
                self.state = 1
                state_changed = True
            # otherwise remain waiting (prefer starting from center)
        elif self.state == 1:
            # Wait for return to center after left
            if abs(recent_mean_yaw) <= self.center_threshold:
                self.state = 2
                state_changed = True
        elif self.state == 2:
            # After returning to center from left, wait for right
            if recent_mean_yaw >= self.yaw_threshold:
                self.state = 3
                state_changed = True
        elif self.state == 3:
            # After right, wait for return to center to count a rep
            if abs(recent_mean_yaw) <= self.center_threshold:
                rep_detected = True
                self.state = 0
                state_changed = True

        # Update state timing
        if state_changed:
            self.last_state_change = len(self.buffer)
            self.state_time = 0
        else:
            self.state_time = len(self.buffer) - self.last_state_change

        return rep_detected

    def get_state_info(self):
        """Return detailed state information for feedback"""
        state_names = {
            0: "Center (waiting)",
            1: "Left position",
            2: "Center (after left)",
            3: "Right position"
        }
        
        return {
            "state": self.state,
            "state_name": state_names.get(self.state, "Unknown"),
            "state_time": self.state_time,
            "progress": min(self.state / 3.0, 1.0)  # Progress as a fraction 0-1
        }


class FlexionExtensionDetector:
    """
    Detects a full flexion-extension rep defined as: center → up → center → down → center

    Assumptions:
    - Angles provided in the frame buffer are already relative to the user's baseline.
    - Pitch is used for up/down/center detection (positive = up, negative = down).
    - Yaw and roll constraints ensure proper form during flexion-extension movements.
    """

    def __init__(self, buffer_size=300, pitch_threshold=13.0, center_threshold=5.0, hold_frames=3,
                 yaw_min=-5.0, yaw_max=5.0, roll_min=-5.0, roll_max=5.0):
        self.buffer_size = buffer_size
        self.pitch_threshold = pitch_threshold
        self.center_threshold = center_threshold
        self.hold_frames = hold_frames

        # Yaw/roll safety window (exclusive lower/upper bounds used in checks)
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.roll_min = roll_min
        self.roll_max = roll_max

        # State machine:
        # 0 = waiting (at/near center)
        # 1 = up detected (pitch >= pitch_threshold)
        # 2 = returned to center after up
        # 3 = down detected (pitch <= -pitch_threshold)
        # When we return to center from state 3, we count a rep and reset to 0
        self.state = 0
        self.buffer = []
        self.state_time = 0  # Time spent in current state for progress tracking
        self.last_state_change = 0  # Time of last state change

    def detect_rep(self, frame_buffer):
        """Update state machine using the most recent frames and return True when a rep completes.

        This function enforces yaw/roll constraints: it computes the recent mean
        yaw and roll and if either value is outside the configured window, the detector 
        will reset to state 0 and return False (no rep counted).
        """
        if not frame_buffer:
            return False

        # Keep buffer within size limit
        self.buffer = frame_buffer[-self.buffer_size:]

        # Extract arrays for axes (assume frames contain 'yaw','pitch','roll')
        yaws = np.array([frame.get("yaw", 0.0) for frame in self.buffer])
        pitches = np.array([frame["pitch"] for frame in self.buffer])
        rolls = np.array([frame.get("roll", 0.0) for frame in self.buffer])

        # Use the mean of the last few frames to reduce jitter
        n = min(self.hold_frames, len(pitches))
        recent_mean_yaw = float(np.mean(yaws[-n:]))
        recent_mean_pitch = float(np.mean(pitches[-n:]))
        recent_mean_roll = float(np.mean(rolls[-n:]))

        # Enforce yaw/roll constraints: if violated, reset detector and do not progress
        yaw_ok = (self.yaw_min < recent_mean_yaw < self.yaw_max)
        roll_ok = (self.roll_min < recent_mean_roll < self.roll_max)
        if not (yaw_ok and roll_ok):
            # Reset to waiting center state to avoid partial sequences being counted
            if self.state != 0:
                self.last_state_change = len(self.buffer)
                print(f"⚠️  Reset due to constraint violation: yaw={recent_mean_yaw:.1f}° (range: {self.yaw_min:.1f} to {self.yaw_max:.1f}), roll={recent_mean_roll:.1f}° (range: {self.roll_min:.1f} to {self.roll_max:.1f})")
            self.state = 0
            self.state_time = 0
            return False

        rep_detected = False
        state_changed = False

        if self.state == 0:
            # If user is already looking up significantly, allow immediate transition
            if recent_mean_pitch >= self.pitch_threshold:
                self.state = 1
                state_changed = True
                print(f"🔄 Up movement detected (pitch: {recent_mean_pitch:.1f}°)")
            # otherwise remain waiting (prefer starting from center)
        elif self.state == 1:
            # Wait for return to center after up
            if abs(recent_mean_pitch) <= self.center_threshold:
                self.state = 2
                state_changed = True
                print(f"🔄 Returned to center after up (pitch: {recent_mean_pitch:.1f}°)")
        elif self.state == 2:
            # After returning to center from up, wait for down
            if recent_mean_pitch <= -self.pitch_threshold:
                self.state = 3
                state_changed = True
                print(f"🔄 Down movement detected (pitch: {recent_mean_pitch:.1f}°)")
        elif self.state == 3:
            # After down, wait for return to center to count a rep
            if abs(recent_mean_pitch) <= self.center_threshold:
                rep_detected = True
                self.state = 0
                state_changed = True
                print(f"✅ Flexion-Extension repetition completed! (pitch: {recent_mean_pitch:.1f}°)")

        # Update state timing
        if state_changed:
            self.last_state_change = len(self.buffer)
            self.state_time = 0
        else:
            self.state_time = len(self.buffer) - self.last_state_change

        return rep_detected

    def get_state_info(self):
        """Return detailed state information for feedback"""
        state_names = {
            0: "Center (waiting)",
            1: "Up position",
            2: "Center (after up)",
            3: "Down position"
        }
        
        return {
            "state": self.state,
            "state_name": state_names.get(self.state, "Unknown"),
            "state_time": self.state_time,
            "progress": min(self.state / 3.0, 1.0)  # Progress as a fraction 0-1
        }