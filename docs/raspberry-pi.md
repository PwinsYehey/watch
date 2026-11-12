# WATCH on a Raspberry Pi

How to run the live monitor on a Raspberry Pi with a camera module and watch it from a browser on a laptop. The Pi needs no screen.

Tested on a Raspberry Pi 4 Model B (8 GB) with Raspberry Pi OS based on Debian 13 (trixie) and a Camera Module 3 NoIR. The monitor runs at about 10 to 15 FPS.

## Hardware

| Part | Details | What it does |
|---|---|---|
| Raspberry Pi 4 Model B | 8 GB RAM | Runs the monitor, the models and the web page. |
| microSD card | 32 GB | Holds the OS, the project, the Python environment and the session reports. |
| Camera Module 3 NoIR | `imx708_noir`, 12 MP, autofocus, no IR filter | The camera in use. The Pi detects it by itself. |
| Camera ribbon | Same width at both ends | Connects the camera to the **CAMERA** connector. Ribbons with one narrow end are for a Pi 5 or Zero and do not fit a Pi 4 directly. |
| USB-C power supply | 5 V 3 A wall adapter | A laptop USB port may not supply enough, and a weak supply can make the Pi reboot or drop off the network. |
| Heatsink or small fan | Recommended | The Pi reached about 71 C under load with the IR modules on. It starts slowing down at 80 C. |
| Ethernet cable | Optional, with a USB adapter if the laptop has no port | Connects the Pi straight to a laptop when there is no shared Wi-Fi. |

### Other cameras that were tried

- **IMX219 night vision camera** (no IR filter, with two IR LED modules). The Pi detected it but never received a picture ("Camera frontend has timed out"), with two different ribbons and the IR modules removed, and later stopped detecting it at all. The board is faulty. A working one needs the IMX219 setting in [Connect the camera](#1-connect-the-camera).
- **A regular IMX219** (with an IR filter). It works in a lit room, but its IR filter blocks the IR LEDs' light, so it sees nothing in the dark. Only a NoIR or night vision camera can use IR light.

### IR LEDs on the Camera Module 3

IR LED boards bought for night vision cameras get no power when screwed onto this camera, because the official module does not put power on its screw holes. They need their own wires from the Pi's 3.3 V and ground pins, and the board and camera should be checked with a multimeter first so the + pad cannot short to the camera. This wiring has not been done or tested.

### Which light it is for

The IR modules light the face without bothering the person, and a camera with no IR filter can see that light. Eyes were visible through a pair of tinted shades this way, but the monitor has not yet been tested reading eyes through shades or a tinted visor. See [Known limits](#known-limits).

## Detection modes

Set with `mode` under `live` in `configs/config.yaml`, then restart the service.

| Mode | What decides | When to use it |
|---|---|---|
| `both` (default) | The face (eyes closed, yawning, head drooped) and body language (posture, nodding, leaning forward, stillness), together. | Works with and without a helmet. When a helmet hides the face, the face rows show "not visible" and the body keeps deciding. |
| `body` | Body language alone. The eye and yawn models are not loaded and there are no check-in prompts. | Users who always wear a helmet. Lighter on the Pi, so the video is smoother. |
| `face` | The face, with body language only once the face is covered. | The original behaviour, before body language was added. |

### Body language signs

All are measured against the sitting posture taken during calibration, so the camera needs the head and both shoulders in view.

| Sign | What it means | Setting under `body` |
|---|---|---|
| Slumped | The shoulders drop, the shoulder line tilts, or the head sinks toward the shoulders. A raised arm tilts the shoulders, so the tilt is ignored while an arm is up. | `drop_limit`, `lean_limit_deg`, `head_drop_limit` |
| Leaning forward | The shoulders look wider because the person has leaned toward the camera. | `forward_limit` |
| Nodding off | The head drops and comes back up quickly, 3 times within 60 seconds. A drop that stays down counts as slumped instead. | `nod_drop`, `nod_max_sec`, `nods_window_sec`, `nods_limit` |
| Not moving | The shoulders and head stay almost still for 45 seconds. | `still_window_sec`, `still_limit` |

The head position is the average of whichever head points the body tracker can see, so it still works under a helmet. The **Body** row on the dashboard shows "slumped", "nodding" or "not moving" when one of them fires, and "arm raised" when an arm is up.

These limits are starting values. They still need tuning with real tests of people wearing helmets.

## Quick test (Pi already set up)

1. Plug in the Pi's power and wait about a minute. The monitor starts by itself.
2. Connect to it one of two ways:
   - **Same Wi-Fi** as the Pi, or
   - **Ethernet cable** straight from the Pi to the laptop. No router or laptop setup is needed; allow up to a minute after plugging in.
3. Open **http://raspberrypi.local:8766** in a browser. If the name is not found, use the Pi's address: `http://<address>:8766`.
4. Fix the camera in place, level with the face and about 1 metre away, with the head and both shoulders in the picture. Too close, and the body tracker cannot find the shoulders.
5. Sit upright, look straight at the camera with your eyes open and hold still for the 4-second calibration.
6. Close your eyes, yawn, drop your head, slump or nod to see WARNING and CRITICAL.
7. When you are done, click **Shut down device**, confirm, wait for the green light to stop blinking, then unplug the power.

To calibrate again, move out of view for at least 3 seconds and come back. Do this whenever the camera has moved, because the head limits depend on where the camera is.

On the page:

| Control | Key | What it does |
|---|---|---|
| Enable alarm sound | | Lets the browser beep while CRITICAL. Browsers only play sound after a click, so press it once per page load. |
| Mute / unmute alarm | `M` | Silences the alarm. |
| Export report | `E` | Saves a PNG graph and CSV log on the Pi in `~/watch/results/sessions/`. |
| Stop monitor | `Q` | Stops the monitor. When it runs as a service it restarts after 5 seconds and recalibrates. |
| Shut down device | | Powers the Pi off after a confirmation. Only shown when the monitor runs with `--allow-shutdown`. |

### Reading the alerts

- One event, such as a yawn lasting 2 seconds, gives a WARNING. The warning stays for 30 seconds after the event, even once the mouth has closed, so that repeats can add up. The **Yawn** row under Signs shows what the mouth is doing now.
- A repeat, a second sign, or closed eyes or a dropped head that persist make it CRITICAL. CRITICAL drops to a "recovering" warning after 5 calm seconds.
- If nobody has been in view for 5 seconds, any alert is cleared and the alarm stops (`lost_reset_sec` under `live`), so an alert cannot stay on when the camera loses the person. When someone comes back into view it calibrates again.
- These times are `window_sec` and `calm_sec` in `configs/config.yaml`.

### Turning it off

Unplugging the Pi while it is running can corrupt the SD card, so shut it down first.

Click **Shut down device** on the page and confirm, wait for the green light to stop blinking (about 15 seconds), then unplug the power. The monitor starts again by itself the next time the Pi is powered on.

If the button is missing or reports that it could not shut down, do it from a terminal on the laptop instead, with the Pi's username in place of `<user>`:

```bash
ssh -t <user>@raspberrypi.local "sudo shutdown -h now"
```

Type the Pi's password when asked, then wait for the green light as above.

## What changed in the code

| File | Change |
|---|---|
| `src/watch/camera.py` | New. `--camera pi` reads a Pi camera module through `rpicam-vid`, since OpenCV cannot open it by index. It uses the camera's default exposure, metering and colour. |
| `src/watch/stream.py` | New. `--stream [PORT]` serves the monitor window to a browser as an MJPEG stream, with the keys and alarm sent through the page. |
| `src/watch/live.py` | Uses the two modules above instead of calling `cv2.VideoCapture`, `cv2.imshow` and `cv2.waitKey` directly. An open eye shape now overrules the eye model unless there is glare, because the model called open eyes closed on this camera's soft IR picture. This applies to every camera. |
| `src/watch/__main__.py` | Adds `--camera pi`, `--stream [PORT]` and `--allow-shutdown` to the `live` action. |
| `scripts/setup-pi.sh` | New. Installs the runtime dependencies on a Pi, without the training extras. |
| `scripts/watch-pi.service` | New. systemd user service that starts the monitor at boot. |
| `scripts/watch-pi-poweroff.rules` | New. polkit rule that lets the **Shut down device** button power the Pi off without a password. |
| `tests/test_stream.py` | New. Tests for the browser page's endpoints, with the power-off command mocked. |
| `src/watch/body.py` | Adds leaning forward, and finds the head from whichever head points are visible. Includes the raised-arm check from `development`. |
| `src/watch/alertness.py` | Adds the nodding counter and the stillness check, and their alert names. |
| `src/watch/live.py` | Adds the `both`, `body` and `face` modes, clears an alert once nobody has been in view for 5 seconds, and runs body tracking every second frame in body mode. The dashboard keeps the same layout in every mode. |
| `scripts/setup-pi.sh` | Also downloads MediaPipe's body pose model, since the monitor cannot download it later on a Pi with no internet. |

Without the new options, `live` opens an OpenCV camera with `--camera N` and shows a desktop window as before.

## First-time setup

Commands marked **Pi** run on the Pi over SSH (`ssh <user>@raspberrypi.local`). Commands marked **laptop** run in a terminal on the laptop.

### 1. Connect the camera

1. Shut the Pi down and unplug its power. Never connect the ribbon while it is powered.
2. On a Pi 4, use the **CAMERA** connector between the HDMI ports and the audio jack, not **DISPLAY**.
3. Lift the connector's clip by its two ends, slide the ribbon in with the metal contacts facing the HDMI ports, and push the clip back down.
4. Do the same at the camera's connector, with the blue strip facing away from the board. Check that the ribbon is straight and does not slide out with a light tug.
5. For an IMX219 camera, the Pi may not detect the board by itself, so tell it which camera is connected. **Pi** (asks for the sudo password, then reboots):

   ```bash
   sudo sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' /boot/firmware/config.txt && echo dtoverlay=imx219 | sudo tee -a /boot/firmware/config.txt && sudo reboot
   ```

   Skip this step for a Camera Module 3. To switch back to one later, undo it and reboot:

   ```bash
   sudo sed -i -e 's/^camera_auto_detect=0/camera_auto_detect=1/' -e '/^dtoverlay=imx219$/d' /boot/firmware/config.txt && sudo reboot
   ```

6. Power the Pi on and check that it sees the camera. **Pi:**

   ```bash
   rpicam-hello --list-cameras
   ```

   An IMX219 camera is listed as `imx219`, and a Camera Module 3 NoIR as `imx708_noir`.

### 2. Copy the project to the Pi

The repository is private, so the Pi cannot clone it without a GitHub login. Copying it from a laptop that already has a checkout avoids putting credentials on the Pi. **Laptop**, inside the checkout:

```bash
git archive --format=tar.gz -o watch.tar.gz HEAD
```

```bash
scp watch.tar.gz <user>@raspberrypi.local:~
```

**Pi:**

```bash
mkdir -p ~/watch && tar xzf ~/watch.tar.gz -C ~/watch && rm ~/watch.tar.gz
```

`git archive` only includes committed files, so commit or check out the branch you want first.

### 3. Install

**Pi:**

```bash
cd ~/watch && ./scripts/setup-pi.sh
```

This installs Python 3.12 through uv and about 1.5 GB of packages, so it takes a while on a slow connection. It also downloads MediaPipe's body pose model, so the Pi needs internet for this step only. The Pi's own Python (3.13 on trixie) is too new for the pinned TensorFlow and MediaPipe versions, which is why the script brings its own. It ends by printing the result of `check`.

### 4. Run it once by hand

**Pi:**

```bash
cd ~/watch && ./scripts/live.sh --camera pi --stream
```

Open http://raspberrypi.local:8766 on the laptop. Press `Ctrl+C` on the Pi to stop.

### 5. Start it at boot

**Pi:**

```bash
mkdir -p ~/.config/systemd/user && cp ~/watch/scripts/watch-pi.service ~/.config/systemd/user/watch.service
```

```bash
systemctl --user enable --now watch.service && loginctl enable-linger
```

The service expects the checkout at `~/watch`.

The service runs the monitor with `--allow-shutdown`, which adds the **Shut down device** button to the page. Powering off normally needs a password, so the button only works once this rule is installed. **Pi** (asks for the sudo password):

```bash
sudo cp ~/watch/scripts/watch-pi-poweroff.rules /etc/polkit-1/rules.d/50-watch-poweroff.rules
```

The rule lets members of the `sudo` group power the machine off without a password. Without it the button reports that it could not shut down.

### 6. Allow a direct Ethernet cable (optional)

By default the Pi's Ethernet port waits for a router to give it an address, so a cable straight to a laptop does not work. This makes the Pi also take a link-local address. **Pi** (asks for the sudo password):

```bash
sudo nmcli con mod netplan-eth0 ipv4.link-local enabled ipv4.dhcp-timeout infinity ipv4.may-fail yes && sudo nmcli con up netplan-eth0
```

`netplan-eth0` is the connection name on Raspberry Pi OS trixie. List the names with `nmcli con show` if it differs: with a USB Ethernet adapter on the Pi it was `Wired connection 2`.

Without this the connection keeps resetting while it waits for a router, which cuts SSH and the page every few minutes, and the Pi has only an IPv6 address, which Edge and Chrome cannot open. With it, the Pi also has a `169.254.x.x` address, shown by `hostname -I`, and `http://169.254.x.x:8766` works in any browser.

On the laptop, IPv6 must be on for the Ethernet adapter, or `raspberrypi.local` is not found over the cable. In Windows: **Win + R**, `ncpa.cpl`, right-click the adapter, **Properties**, tick **Internet Protocol Version 6 (TCP/IPv6)**.

## Commands

All on the **Pi**.

| Task | Command |
|---|---|
| Is the monitor running? | `systemctl --user status watch.service` |
| Restart it | `systemctl --user restart watch.service` |
| Stop it until the next boot | `systemctl --user stop watch.service` |
| Start it again | `systemctl --user start watch.service` |
| Stop it starting at boot | `systemctl --user disable watch.service` |
| Follow its log | `journalctl --user-unit watch.service -f` |
| Run it by hand (stop the service first) | `cd ~/watch && ./scripts/live.sh --camera pi --stream` |
| Use another port | `./scripts/live.sh --camera pi --stream 9000` |
| Check the install | `cd ~/watch && .venv/bin/python -m watch check` |
| Run the unit tests | `cd ~/watch && ./scripts/test.sh` |
| Check the camera | `rpicam-hello --list-cameras` |
| Temperature and power warnings | `vcgencmd measure_temp; vcgencmd get_throttled` |
| Shut down safely | `sudo shutdown now` |

Only one program can use the camera at a time, so stop the service before running `live` by hand or using `rpicam-hello`.

## Changing the picture

The camera runs on its own default exposure, metering and colour, with no extra filters. To change the picture, add `rpicam-vid` options such as `--ev`, `--saturation` or `--metering` to the command in `src/watch/camera.py`, then restart the service. On a NoIR camera the default colour looks pink or purple, which does not matter for body tracking; `--saturation 0` makes it black and white.

## Training on your own recordings

Behind tinted lenses the eye shape is a guess, so the eye model has to decide. These steps teach it your lenses and IR light. Stop the service first, since the camera can be used by one program at a time.

1. Record a take per person and per condition, such as with and without sunglasses. The buzzer cues each part: one beep, eyes open; two beeps, eyes closed; a long beep, done. **Pi:**

   ```bash
   cd ~/watch && .venv/bin/python -m watch record yeri shades1 --camera pi
   ```

2. With at least two takes, fine-tune. A quarter of the takes are held out, and the model is replaced only if those score higher. The original stays as `models/eye_model.base.keras`. **Pi:**

   ```bash
   cd ~/watch && .venv/bin/python -m watch finetune
   ```

3. If the held-out closed-eye recall improved, set `eye_decision: cnn` under `live` in `configs/config.yaml` so the model decides on its own, then start the service again.

Calibration ignores eyes it cannot see clearly (`min_open_ear` under `calibration`), so starting with sunglasses on keeps the default eye thresholds instead of learning wrong ones.

## Updating the code on the Pi

Repeat step 2 to copy the new version over, then restart. **Pi:**

```bash
systemctl --user restart watch.service
```

Run `./scripts/setup-pi.sh` again only if `pyproject.toml` changed.

## Troubleshooting

| Problem | What to check |
|---|---|
| The page does not load | Give the Pi a minute after power-on. Check `systemctl --user status watch.service`. Over a cable, wait up to a minute after plugging in. |
| The Pi freezes (only the red light on, green stopped blinking) | Unplug and replug it. It froze when body tracking ran on every frame; body mode now runs it every second frame (`body_mode_every_n_frames`). If it happens again, raise that number or add cooling. |
| "Camera frontend has timed out" | The Pi talks to the camera but receives no picture. Reseat the ribbon and try another ribbon; if it continues, the camera board is faulty. |
| `raspberrypi.local` is not found | The laptop may block mDNS (some managed machines and firewalls do). Find the Pi's address with `hostname -I` on the Pi and use `http://<address>:8766`. |
| The monitor keeps restarting with `Temporary failure in name resolution` in its log | MediaPipe is trying to download its body pose model with no internet. Run `./scripts/setup-pi.sh` once with internet, or copy `pose_landmark_lite.tflite` from <https://storage.googleapis.com/mediapipe-assets/> into `.venv/lib/python3.12/site-packages/mediapipe/modules/pose_landmark/`. |
| `No cameras available!` or `Could not open camera pi` | The page stays down because the monitor keeps restarting without a camera. Reseat the ribbon at both ends with the Pi powered off. For an IMX219 camera, apply the setting in step 1. |
| Black video, "NO FACE DETECTED" | The lens is covered or pointing away, or the room is too dark. A camera with an IR filter stays black in the dark even with IR LEDs on. |
| "Person: shoulders not seen" | The camera is too close or aimed too high. Move it back so the head and both shoulders are in view. |
| The picture is pink or purple | Normal for a NoIR camera. Add `--saturation 0` to the command in `src/watch/camera.py` for black and white. |
| The face is washed out and "Eyes: 0 of 2 readable" | Too much IR on the face. Move back or angle the IR modules. |
| "Lighting: low, boosted" | Add light on the face. A lamp from the front or a window you are facing works. |
| "Sharpness: blurry" | Sit 50 to 70 cm back and hold still. If it stays soft, power off and try a very small twist of the lens if it turns easily; some are glued, so do not force it. |
| "CAMERA LOST" | No new picture arrived for 2 seconds. The ribbon may be loose, another program may be using the camera, or the IR modules may be drawing too much power. Reseat the ribbon with the Pi powered off, then try one IR module. |
| It says the head is drooped while upright | It calibrated while the camera was moving. Fix the camera in place and calibrate again. |
| The Pi reboots or drops off the network under load | Power it from a 5 V 3 A supply rather than a laptop USB port. `vcgencmd get_throttled` shows anything other than `0x0` after an undervoltage. |
| The Pi gets too hot | Add a heatsink or fan, keep air moving under the board and run shorter tests. |
| No alarm sound | Click **Enable alarm sound** on the page; it has to be clicked again after a reload. |
| It keeps recalibrating | The face or camera was lost for 3 seconds or more (`recalibrate_gap_sec` in `configs/config.yaml`). Mount the camera so the face stays in frame. |

## Known limits

- **The body language limits are untested starting values.** They need tuning with people wearing helmets in the real seating position.
- **About 10 to 15 FPS on a Pi 4.** Blinks shorter than about 0.1 s can fall between frames.
- **Not finished for a tinted visor.** Eyes were visible through tinted shades on the camera, but the monitor has not been tested reading them through shades or a visor.
- **Daytime use is the current target.** Working in full darkness has not been tested.
- **The open-eye change has not been tested on a face yet.** The change in `live.py` passes the unit tests, but no one has confirmed yet that it stops the false alarm while still catching closed eyes.
- **The Pi runs hot** with no cooling, so keep sessions short until a heatsink or fan is fitted.
- Alert timing is measured in seconds, so the logic still holds, but the video is not smooth. Running the TFLite models in `models/` instead of the Keras ones should be faster and has not been done yet.
- The page has no login. Anyone on the same network can view the camera and press the buttons, including **Shut down device**, so use it on a trusted network only.
- Exported reports stay on the Pi; copy them off with `scp <user>@raspberrypi.local:~/watch/results/sessions/* .`.
- `--stream` should also work on Windows and Linux with a normal webcam, but it has only been run on the Pi.
