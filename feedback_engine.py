import cv2
import numpy as np
import time

class FeedbackEngine:
    def __init__(self, tolerance_base=4, angle_tolerance=10, yaw_threshold=15.0, center_threshold=5.0, enable_audio=True):
        self.tolerance_base = tolerance_base
        self.angle_tolerance = angle_tolerance
        self.yaw_threshold = yaw_threshold
        self.center_threshold = center_threshold
        self.last_feedback_time = 0
        self.feedback_cooldown = 2.0  # seconds between feedback messages
        self.enable_audio = enable_audio
        self.last_audio_feedback = ""
        self.last_audio_time = 0
        
        # Rep completion animation
        self.rep_animation_start_time = 0
        self.rep_animation_duration = 2.0  # Animation duration in seconds
        self.current_rep_animated = 0
        self.show_rep_animation = False
        
        # Try to import text-to-speech for audio feedback
        self.tts_available = False
        if enable_audio:
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)  # Speed of speech
                self.tts_engine.setProperty('volume', 0.7)  # Volume level
                self.tts_available = True
            except ImportError:
                print("Note: pyttsx3 not available. Audio feedback disabled. Install with: pip install pyttsx3")
                self.tts_available = False

    def display_feedback(self, frame, metrics, current_angles, ref_angles=None, rep_count=0, rep_state=0):
        import time
        
        # Handle both comparison metrics and live tracking
        if "status" in metrics and metrics["status"] == "live_tracking":
            mean_error = 0
            time_ratio = 1
            status = "Live tracking"
            color = (0, 255, 0)
        else:
            mean_error = metrics.get("mean_error", 0)
            time_ratio = metrics.get("time_ratio", 1)
            tolerance = max(3 * mean_error, self.tolerance_base)
            status = "Good, match" if mean_error <= tolerance else "Adjust rotation"
            color = (0, 255, 0) if mean_error <= tolerance else (0, 0, 255) if mean_error > 2 * tolerance else (0, 165, 255)

        timing = ""
        if time_ratio > 1.2:
            timing = "Move faster"
        elif time_ratio < 0.8:
            timing = "Move slower"
        else:
            timing = "Good pace"

        # Enhanced rotation feedback based on current state and thresholds
        current_time = time.time()
        rotation_feedback = self._get_rotation_feedback(current_angles, rep_state)
        
        rotation_status = "N/A"
        rotation_color = (255, 255, 255)
        
        if ref_angles is not None:
            angle_diff = np.abs(current_angles - ref_angles)
            if np.all(angle_diff < self.angle_tolerance):
                rotation_status = "Correct rotation"
                rotation_color = (0, 255, 0)
            else:
                rotation_status = "Incorrect rotation"
                rotation_color = (0, 0, 255)
        else:
            # Use live feedback when no reference is available
            rotation_status = rotation_feedback["status"]
            rotation_color = rotation_feedback["color"]

        h, w = frame.shape[:2]
        # For dual display, position feedback on the left half (live feed area)
        live_feed_width = w // 2  # Live feed is left half of combined frame
        
        # Expanded box to accommodate more feedback text
        box_w = min(420, int(live_feed_width * 0.9))
        box_h = 280
        box_x = 10
        box_y = 60  # Position below the "LIVE FEED" title
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)

        # Text positions inside box with enough spacing to avoid overlap
        text_x = box_x + 10
        y = box_y + 25
        line_h = 25
        
        cv2.putText(frame, f"Status: {status}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += line_h
        cv2.putText(frame, f"Timing: {timing}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += line_h
        cv2.putText(frame, f"Rotation: {rotation_status}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rotation_color, 1)
        y += line_h
        
        # Show specific guidance based on rotation feedback
        if rotation_feedback["guidance"]:
            guidance_lines = rotation_feedback["guidance"].split('\n')
            for guidance_line in guidance_lines:
                if guidance_line.strip():
                    cv2.putText(frame, guidance_line.strip(), (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y += line_h - 5
            
            # Provide audio feedback for guidance
            self._provide_audio_feedback(rotation_feedback["guidance"])
        
        y += 5  # Add some spacing
        cv2.putText(frame, f"Yaw: {current_angles[0]:.1f}°", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += line_h - 5
        cv2.putText(frame, f"Pitch: {current_angles[1]:.1f}°", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += line_h - 5
        cv2.putText(frame, f"Roll: {current_angles[2]:.1f}°", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw compact bars at the bottom-right of the live feed area
        bar_width = 100
        bar_height = 10
        max_angle = 60
        bar_base_y = h - 40
        bar_x = live_feed_width - 140  # Position within live feed area
        for i, (angle, name) in enumerate(zip(current_angles, ["Yaw", "Pitch", "Roll"])):
            bar_length = int(bar_width * min(abs(angle) / max_angle, 1))
            bar_y = bar_base_y - i * (bar_height + 8)
            # background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            # fill
            fill_color = (0, 255, 0) if abs(angle) < self.angle_tolerance else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), fill_color, -1)
            cv2.putText(frame, name, (bar_x - 60, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw rep count top-right
        rep_text = f"Reps: {rep_count}"
        cv2.putText(frame, rep_text, (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw rep progress indicator (center -> left -> center -> right -> center)
        # We'll show 5 small circles horizontally indicating the phases and highlight the current one
        phases = ["C", "L", "C", "R", "C"]
        phase_colors = {
            0: (200, 200, 200),
            1: (200, 200, 200),
            2: (200, 200, 200),
            3: (200, 200, 200),
            4: (200, 200, 200),
        }
        # Map rep_state to phase index. Our RepDetector states: 0(waiting center),1(left),2(center after left),3(right)
        state_to_phase = {0:0, 1:1, 2:2, 3:3}
        active_idx = state_to_phase.get(rep_state, 0)
        # center after right maps back to final center (index 4) if we want to show it briefly
        # Draw at top-center area of the live feed (left half of dual display)
        live_feed_width = w // 2
        center_x = live_feed_width // 2  # Center of the live feed area
        px = center_x - 80
        py = 100  # Position below the feedback box
        radius = 12
        spacing = 40
        for i, label in enumerate(phases):
            x = px + i * spacing
            color = (0, 255, 0) if i == active_idx else (100, 100, 100)
            cv2.circle(frame, (x, py), radius, color, -1)
            cv2.putText(frame, label, (x - 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def display_summary(self, rep_count, metrics):
        # Handle both comparison metrics and live tracking status
        if "status" in metrics and metrics["status"] == "live_tracking":
            print(f"Rep {rep_count} completed - Live tracking mode")
        else:
            print(f"Rep {rep_count}:")
            print(f"  Mean Error: {metrics.get('mean_error', 0):.1f}°")
            print(f"  Max Error: {metrics.get('max_error', 0):.1f}°")
            print(f"  Time Ratio: {metrics.get('time_ratio', 1):.2f}")

    def _get_rotation_feedback(self, current_angles, rep_state):
        """
        Provides specific feedback based on current angles and expected rotation state.
        
        Args:
            current_angles: [yaw, pitch, roll] current head angles
            rep_state: Current state from RepDetector (0=center, 1=left, 2=center after left, 3=right)
        
        Returns:
            dict with keys: status, color, guidance
        """
        yaw, pitch, roll = current_angles
        
        # Define expected states and provide appropriate feedback
        feedback = {
            "status": "Good position",
            "color": (0, 255, 0),
            "guidance": ""
        }
        
        # Check pitch and roll constraints first (these should always be maintained)
        pitch_ok = -5.0 < pitch < 10.0
        roll_ok = -10.0 < roll < 10.0
        
        if not pitch_ok or not roll_ok:
            feedback["status"] = "Adjust head position"
            feedback["color"] = (0, 0, 255)
            guidance_parts = []
            
            if not pitch_ok:
                if pitch <= -5.0:
                    guidance_parts.append("Lift your chin up")
                elif pitch >= 10.0:
                    guidance_parts.append("Lower your chin down")
            
            if not roll_ok:
                if roll <= -10.0:
                    guidance_parts.append("Tilt head less to the right")
                elif roll >= 10.0:
                    guidance_parts.append("Tilt head less to the left")
            
            feedback["guidance"] = "\n".join(guidance_parts)
            return feedback
        
        # State-specific feedback for yaw rotations
        if rep_state == 0:  # Waiting at center
            if abs(yaw) <= self.center_threshold:
                feedback["status"] = "Ready - Turn left to start"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Keep head centered, then turn left"
            elif yaw < -self.center_threshold:
                if yaw <= -self.yaw_threshold:
                    feedback["status"] = "Good! Now return to center"
                    feedback["color"] = (0, 255, 0)
                    feedback["guidance"] = "Turn head back to center"
                else:
                    feedback["status"] = "Turn more to the left"
                    feedback["color"] = (255, 165, 0)
                    feedback["guidance"] = f"Need {self.yaw_threshold - abs(yaw):.1f}° more left"
            elif yaw > self.center_threshold:
                feedback["status"] = "Return to center first"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = "Turn head back to center position"
                
        elif rep_state == 1:  # Left position detected
            if yaw <= -self.yaw_threshold:
                feedback["status"] = "Good left position!"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Now return to center"
            else:
                feedback["status"] = "Turn more to the left"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = f"Need {self.yaw_threshold - abs(yaw):.1f}° more left"
                
        elif rep_state == 2:  # Center after left
            if abs(yaw) <= self.center_threshold:
                feedback["status"] = "Good center! Now turn right"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Turn head to the right now"
            elif yaw < 0:
                feedback["status"] = "Return to center"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = "Turn head back to center first"
            elif yaw > self.center_threshold:
                if yaw >= self.yaw_threshold:
                    feedback["status"] = "Good! Now return to center"
                    feedback["color"] = (0, 255, 0)
                    feedback["guidance"] = "Turn head back to center"
                else:
                    feedback["status"] = "Turn more to the right"
                    feedback["color"] = (255, 165, 0)
                    feedback["guidance"] = f"Need {self.yaw_threshold - yaw:.1f}° more right"
                    
        elif rep_state == 3:  # Right position detected
            if yaw >= self.yaw_threshold:
                feedback["status"] = "Good right position!"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Now return to center to complete rep"
            else:
                feedback["status"] = "Turn more to the right"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = f"Need {self.yaw_threshold - yaw:.1f}° more right"
        
        return feedback

    def _provide_audio_feedback(self, guidance_text):
        """
        Provides audio feedback if enabled and available.
        Uses cooldown to prevent spam.
        """
        if not self.tts_available or not self.enable_audio:
            return
            
        current_time = time.time()
        
        # Only provide audio feedback if enough time has passed and the message is different
        if (current_time - self.last_audio_time > self.feedback_cooldown and 
            guidance_text != self.last_audio_feedback and 
            guidance_text.strip()):
            
            try:
                # Extract the main instruction from guidance (first line)
                main_instruction = guidance_text.split('\n')[0].strip()
                if main_instruction:
                    self.tts_engine.say(main_instruction)
                    self.tts_engine.runAndWait()
                    self.last_audio_feedback = guidance_text
                    self.last_audio_time = current_time
            except Exception as e:
                # Silently handle TTS errors
                pass

    def toggle_audio_feedback(self):
        """Toggle audio feedback on/off"""
        self.enable_audio = not self.enable_audio
        return self.enable_audio

    def set_audio_feedback(self, enabled):
        """Enable or disable audio feedback"""
        self.enable_audio = enabled and self.tts_available

    def display_flexion_extension_feedback(self, frame, metrics, current_angles, rep_count=0, rep_state=0):
        """
        Display feedback for flexion-extension (up-down) exercise
        """
        import time
        
        # Handle flexion-extension specific feedback
        mean_error = 0
        time_ratio = 1
        status = "Flexion-Extension tracking"
        color = (0, 255, 0)

        # Enhanced flexion-extension feedback based on current state and thresholds
        current_time = time.time()
        flexion_feedback = self._get_flexion_extension_feedback(current_angles, rep_state)
        
        rotation_status = flexion_feedback["status"]
        rotation_color = flexion_feedback["color"]

        h, w = frame.shape[:2]
        
        # Expanded box to accommodate more feedback text
        box_w = min(420, int(w * 0.4))
        box_h = 280
        box_x = 10
        box_y = 60  # Position below the title
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)

        # Text positions inside box with enough spacing to avoid overlap
        text_x = box_x + 10
        y = box_y + 25
        line_h = 25
        
        cv2.putText(frame, f"Status: {status}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += line_h
        cv2.putText(frame, f"Movement: {rotation_status}", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rotation_color, 1)
        y += line_h
        
        # Show specific guidance based on flexion-extension feedback
        if flexion_feedback["guidance"]:
            guidance_lines = flexion_feedback["guidance"].split('\n')
            for guidance_line in guidance_lines:
                if guidance_line.strip():
                    cv2.putText(frame, guidance_line.strip(), (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y += line_h - 5
            
            # Provide audio feedback for guidance
            self._provide_audio_feedback(flexion_feedback["guidance"])
        
        y += 5  # Add some spacing
        cv2.putText(frame, f"Yaw: {current_angles[0]:.1f}°", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += line_h - 5
        cv2.putText(frame, f"Pitch: {current_angles[1]:.1f}°", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += line_h - 5
        cv2.putText(frame, f"Roll: {current_angles[2]:.1f}°", (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw compact bars 
        bar_width = 100
        bar_height = 10
        max_angle = 60
        bar_base_y = h - 40
        bar_x = w - 140  
        for i, (angle, name) in enumerate(zip(current_angles, ["Yaw", "Pitch", "Roll"])):
            bar_length = int(bar_width * min(abs(angle) / max_angle, 1))
            bar_y = bar_base_y - i * (bar_height + 8)
            # background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            # fill
            fill_color = (0, 255, 0) if abs(angle) < self.angle_tolerance else (0, 165, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_length, bar_y + bar_height), fill_color, -1)
            cv2.putText(frame, name, (bar_x - 60, bar_y + bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw rep progress indicator (center -> down -> center -> up -> center)
        phases = ["C", "U", "C", "D", "C"]
        state_to_phase = {0:0, 1:1, 2:2, 3:3}
        active_idx = state_to_phase.get(rep_state, 0)
        
        center_x = w // 2
        px = center_x - 80
        py = 130  
        radius = 12
        spacing = 40
        for i, label in enumerate(phases):
            x = px + i * spacing
            color = (0, 255, 0) if i == active_idx else (100, 100, 100)
            cv2.circle(frame, (x, py), radius, color, -1)
            cv2.putText(frame, label, (x - 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _get_flexion_extension_feedback(self, current_angles, rep_state):
        """
        Provides specific feedback based on current angles and expected flexion-extension state.
        
        Args:
            current_angles: [yaw, pitch, roll] current head angles
            rep_state: Current state from FlexionExtensionDetector (0=center, 1=down, 2=center after down, 3=up)
        
        Returns:
            dict with keys: status, color, guidance
        """
        yaw, pitch, roll = current_angles
        
        # Define expected states and provide appropriate feedback
        feedback = {
            "status": "Good position",
            "color": (0, 255, 0),
            "guidance": ""
        }
        
        # Check yaw and roll constraints first (these should always be maintained)
        yaw_ok = -5.0 < yaw < 5.0
        roll_ok = -5.0 < roll < 5.0
        
        if not yaw_ok or not roll_ok:
            feedback["status"] = "Keep head straight"
            feedback["color"] = (0, 0, 255)
            guidance_parts = []
            
            if not yaw_ok:
                if yaw <= -5.0:
                    guidance_parts.append("Turn head back from left")
                elif yaw >= 5.0:
                    guidance_parts.append("Turn head back from right")
            
            if not roll_ok:
                if roll <= -5.0:
                    guidance_parts.append("Stop tilting head right")
                elif roll >= 5.0:
                    guidance_parts.append("Stop tilting head left")
            
            feedback["guidance"] = "\n".join(guidance_parts)
            return feedback
        
        # State-specific feedback for pitch movements
        pitch_threshold = 15.0
        center_threshold = 5.0
        
        if rep_state == 0:  # Waiting at center
            if abs(pitch) <= center_threshold:
                feedback["status"] = "Ready - Look down to start"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Keep head centered, then look down"
            elif pitch < -center_threshold:
                if pitch <= -pitch_threshold:
                    feedback["status"] = "Good! Now return to center"
                    feedback["color"] = (0, 255, 0)
                    feedback["guidance"] = "Lift head back to center"
                else:
                    feedback["status"] = "Look further down"
                    feedback["color"] = (255, 165, 0)
                    feedback["guidance"] = f"Need {pitch_threshold - abs(pitch):.1f}° more down"
            elif pitch > center_threshold:
                feedback["status"] = "Return to center first"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = "Lower head back to center position"
                
        elif rep_state == 1:  # Down position detected
            if pitch <= -pitch_threshold:
                feedback["status"] = "Good down position!"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Now return to center"
            else:
                feedback["status"] = "Look further down"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = f"Need {pitch_threshold - abs(pitch):.1f}° more down"
                
        elif rep_state == 2:  # Center after down
            if abs(pitch) <= center_threshold:
                feedback["status"] = "Good center! Now look up"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Look up towards ceiling now"
            elif pitch < 0:
                feedback["status"] = "Return to center"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = "Lift head back to center first"
            elif pitch > center_threshold:
                if pitch >= pitch_threshold:
                    feedback["status"] = "Good! Now return to center"
                    feedback["color"] = (0, 255, 0)
                    feedback["guidance"] = "Lower head back to center"
                else:
                    feedback["status"] = "Look further up"
                    feedback["color"] = (255, 165, 0)
                    feedback["guidance"] = f"Need {pitch_threshold - pitch:.1f}° more up"
                    
        elif rep_state == 3:  # Up position detected
            if pitch >= pitch_threshold:
                feedback["status"] = "Good up position!"
                feedback["color"] = (0, 255, 0)
                feedback["guidance"] = "Now return to center to complete rep"
            else:
                feedback["status"] = "Look further up"
                feedback["color"] = (255, 165, 0)
                feedback["guidance"] = f"Need {pitch_threshold - pitch:.1f}° more up"
        
        return feedback

    def trigger_rep_animation(self, rep_count):
        """Trigger the rep completion animation"""
        self.rep_animation_start_time = time.time()
        self.current_rep_animated = rep_count
        self.show_rep_animation = True

    def display_rep_animation(self, frame):
        """Display animated rep completion popup"""
        if not self.show_rep_animation:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.rep_animation_start_time
        
        if elapsed_time > self.rep_animation_duration:
            self.show_rep_animation = False
            return
            
        # Animation progress (0 to 1)
        progress = elapsed_time / self.rep_animation_duration
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Animation phases
        if progress < 0.3:  # Grow phase
            scale = progress / 0.3
            alpha = 1.0
        elif progress < 0.7:  # Stay phase
            scale = 1.0
            alpha = 1.0
        else:  # Fade phase
            scale = 1.0 - (progress - 0.7) / 0.3 * 0.3  # Shrink slightly
            alpha = 1.0 - (progress - 0.7) / 0.3
            
        # Calculate sizes based on scale
        circle_radius = int(80 * scale)
        text_size = 3.0 * scale
        text_thickness = int(8 * scale)
        
        # Create overlay for transparency effect
        overlay = frame.copy()
        
        # Draw animated circle background
        cv2.circle(overlay, (center_x, center_y), circle_radius, (0, 255, 0), -1)
        cv2.circle(overlay, (center_x, center_y), circle_radius, (255, 255, 255), 4)
        
        # Draw rep number
        rep_text = str(self.current_rep_animated)
        text_size_cv = cv2.getTextSize(rep_text, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)[0]
        text_x = center_x - text_size_cv[0] // 2
        text_y = center_y + text_size_cv[1] // 2
        cv2.putText(overlay, rep_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), text_thickness)
        
        # Draw "REP COMPLETED!" text below
        if progress < 0.8:  # Show text for most of the animation
            completion_text = "REP COMPLETED!"
            completion_size = 1.2 * scale
            completion_thickness = int(3 * scale)
            completion_text_size = cv2.getTextSize(completion_text, cv2.FONT_HERSHEY_SIMPLEX, completion_size, completion_thickness)[0]
            completion_x = center_x - completion_text_size[0] // 2
            completion_y = center_y + circle_radius + 50
            cv2.putText(overlay, completion_text, (completion_x, completion_y), cv2.FONT_HERSHEY_SIMPLEX, completion_size, (0, 255, 0), completion_thickness)
        
        # Apply transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)