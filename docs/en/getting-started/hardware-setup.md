# Hardware setup

Complete the fake-backend [quickstart](quickstart.md) before connecting a physical
hand. Keep fingers, cables, and tools clear of the mechanism whenever torque may
be enabled.

## Find the serial port

Typical port names are:

- Linux: `/dev/ttyUSB0` or, preferably, a stable `/dev/serial/by-id/...` path
- macOS: `/dev/cu.usbserial-...`

On Ubuntu, the following temporary settings match the existing ghand workflow:

```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
sudo chmod 666 /dev/ttyUSB0
```

The permission change lasts only until the device is reconnected and grants every
local user access to that device. For a persistent installation, use an
administrator-approved device group or udev rule matched to the actual USB serial
adapter instead of copying a vendor/product ID from another system.

## Run the hardware check

With the hand powered and the adapter connected, run:

```bash
uv run ghand doctor
```

It never moves the hand. It verifies the serial adapter, port permissions, the
USB latency timer, all 16 motor IDs, firmware versions, Homing Offset(20) = 0,
supply voltage (12 V ±10%), temperature, and hardware error flags, and exits
non-zero when any row fails. `--port auto` on the other commands selects the
same adapter.

## Read before moving

To inspect one motor by hand, confirm that the bus can read its encoder without
enabling torque:

```bash
uv run ghand motor test \
  --port /dev/serial/by-id/<adapter> \
  --id <motor-id> \
  --encoder-only
```

Before any motion test:

1. Verify the serial path, baud rate, and motor ID.
2. Place the hand where a small unexpected motion cannot trap anything.
3. Make sure power and communication can be disconnected immediately.
4. Leave `--keep-torque` off until the installation has been validated.

Run `uv run ghand motor test --help` before selecting a moving test. The command
uses a small relative move and returns the motor, but the correct safe range still
depends on the assembled hand.

## Open the real-hardware UI

```bash
uv run --extra ui ghand ui --port /dev/serial/by-id/<adapter>
```

Open <http://localhost:8080>. Confirm telemetry and joint mapping before sending
wide joint targets.

## Speed limits

Scripts that use `Hand.connect()` move conservatively by default. Three settings
bound the speed, and each can be changed when connecting:

| Argument | Default | Effect |
| --- | --- | --- |
| `target_velocity_limit` | 2 rad/s | Caps how fast the commanded joint angles advance |
| `profile_velocity`, `profile_acceleration` | 120 and 80, about 2.9 rad/s and 30 rad/s² | Dynamixel Profile Velocity(112) and Profile Acceleration(108) that each servo applies on top; 0 removes the limit |
| `target_hz` | 30 Hz | How often `move_to` sends a new command |

To let the servos move as fast as they can, the way `ghand teleop` runs them:

```python
from ghand import Hand

hand = Hand.connect(
    target_velocity_limit=1000.0,
    profile_velocity=0,
    profile_acceleration=0,
)
```

Two things change at that speed:

- Without a servo profile the hand jumps to every command, so an interpolated
  move looks jerky at 30 Hz. Raise `target_hz`, or keep a profile with a high
  acceleration; `examples/04_pose_sequence.py` does both.
- Joints that cannot keep up with the command stop arriving together, so a fast
  multi-joint move can pass through poses that the slow one avoids. Check such a
  move in simulation or at a lower speed before running it at full speed.

## Continue

- [Web UI guide](../guides/web-ui.md)
- [Teleoperation](../guides/teleoperation.md)
- [Troubleshooting](../troubleshooting.md)
