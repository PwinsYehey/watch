# WATCH on a Raspberry Pi

How to run the live monitor on a Raspberry Pi with a camera module and watch it from a browser on a laptop.

Tested on a Raspberry Pi 4 Model B (8 GB) with Raspberry Pi OS based on Debian 13 (trixie) and a Camera Module 3 NoIR. It runs at about 10 FPS with a face in view.

## Quick test (Pi already set up)

1. Power the Pi and wait about a minute.
2. Connect to it one of two ways:
   - **Same Wi-Fi** as the Pi, or
   - **Ethernet cable** straight from the Pi to the laptop. No router or laptop setup is needed; allow up to a minute after plugging in.
3. Open **http://raspberrypi.local:8766** in a browser.
4. Look at the camera for the 4-second calibration, then close your eyes, yawn or drop your head to see WARNING and CRITICAL.

On the page:

| Control | Key | What it does |
|---|---|---|
| Enable alarm sound | | Lets the browser beep while CRITICAL. Browsers only play sound after a click, so press it once per page load. |
| Mute / unmute alarm | `M` | Silences the alarm. |
| Export report | `E` | Saves a PNG graph and CSV log on the Pi in `~/watch/results/sessions/`. |
| Stop monitor | `Q` | Stops the monitor. When it runs as a service it restarts after 5 seconds and recalibrates. |
| Shut down device | | Powers the Pi off after a confirmation. Only shown when the monitor runs with `--allow-shutdown`. |

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
| `src/watch/camera.py` | New. `--camera pi` reads a Pi camera module through `rpicam-vid`, since OpenCV cannot open it by index. |
| `src/watch/stream.py` | New. `--stream [PORT]` serves the monitor window to a browser as an MJPEG stream, with the keys and alarm sent through the page. |
| `src/watch/live.py` | Uses the two modules above instead of calling `cv2.VideoCapture`, `cv2.imshow` and `cv2.waitKey` directly. |
| `src/watch/__main__.py` | Adds `--camera pi`, `--stream [PORT]` and `--allow-shutdown` to the `live` action. |
| `scripts/setup-pi.sh` | New. Installs the runtime dependencies on a Pi, without the training extras. |
| `scripts/watch-pi.service` | New. systemd user service that starts the monitor at boot. |
| `scripts/watch-pi-poweroff.rules` | New. polkit rule that lets the **Shut down device** button power the Pi off without a password. |
| `tests/test_stream.py` | New. Tests for the browser page's endpoints, with the power-off command mocked. |

Without the new options, `live` behaves as before: `--camera N` opens an OpenCV camera and shows a desktop window.

## First-time setup

Commands marked **Pi** run on the Pi over SSH (`ssh <user>@raspberrypi.local`). Commands marked **laptop** run in a terminal on the laptop.

### 1. Connect the camera

1. Shut the Pi down and unplug its power. Never connect the ribbon while it is powered.
2. On a Pi 4, use the **CAMERA** connector between the HDMI ports and the audio jack, not **DISPLAY**.
3. Lift the connector's clip by its two ends, slide the ribbon in with the metal contacts facing the HDMI ports, and push the clip back down.
4. Power the Pi on and check that it sees the camera. **Pi:**

   ```bash
   rpicam-hello --list-cameras
   ```

   A Camera Module 3 NoIR is listed as `imx708_noir`.

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

This installs Python 3.12 through uv and about 1.5 GB of packages, so it takes a while on a slow connection. The Pi's own Python (3.13 on trixie) is too new for the pinned TensorFlow and MediaPipe versions, which is why the script brings its own. It ends by printing the result of `check`.

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

`netplan-eth0` is the connection name on Raspberry Pi OS trixie; list the names with `nmcli con show` if it differs.

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
| `raspberrypi.local` is not found | The laptop may block mDNS (some managed machines and firewalls do). Find the Pi's address with `hostname -I` on the Pi and use `http://<address>:8766`. |
| Black video, "NO FACE DETECTED" | The lens is covered or pointing away, or the room is too dark. |
| "Lighting: low, boosted" | Add light on the face. A NoIR camera also works with an infrared lamp, including in the dark. |
| "CAMERA LOST" | The ribbon is loose, or another program is using the camera. Reseat it with the Pi powered off. |
| `Could not open camera pi` | `rpicam-hello --list-cameras` should list a camera; if not, reseat the ribbon. |
| The Pi reboots or drops off the network under load | Power it from a 5 V 3 A supply rather than a laptop USB port. `vcgencmd get_throttled` shows anything other than `0x0` after an undervoltage. |
| No alarm sound | Click **Enable alarm sound** on the page; it has to be clicked again after a reload. |
| It keeps recalibrating | The face left the view for 3 seconds or more (`recalibrate_gap_sec` in `configs/config.yaml`). Mount the camera so the face stays in frame. |

## Known limits

- About 10 FPS on a Pi 4. Alert timing is measured in seconds, so the logic still holds, but the video is not smooth. Running the TFLite models in `models/` instead of the Keras ones should be faster and has not been done yet.
- The page has no login. Anyone on the same network can view the camera and press the buttons, including **Shut down device**, so use it on a trusted network only.
- Exported reports stay on the Pi; copy them off with `scp <user>@raspberrypi.local:~/watch/results/sessions/* .`.
- `--stream` should also work on Windows and Linux with a normal webcam, but it has only been run on the Pi.
