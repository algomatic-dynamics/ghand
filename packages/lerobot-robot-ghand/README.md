# lerobot_robot_ghand

Installing this distribution lets the LeRobot command-line tools discover the
LEAP V1 KAIZEN hand automatically, because LeRobot imports every installed
package whose name starts with `lerobot_robot_`:

```bash
lerobot-record --robot.type=ghand --robot.backend=fake ...
```

The implementation lives in `ghand.lerobot_plugin`; this package only re-exports
it and depends on `ghand[lerobot]`.
