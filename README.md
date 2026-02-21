# Kepler ROV

A robotics team based out of Aptos High School. We build an underwater ROV to compete in the MATE ROV competition

## ROV Thruster Simulator

A lightweight visual simulator is available in `rov_simulator.py`.

- Uses the same joystick mixing logic as `top-side.py` for all 8 thrusters.
- Draws the ROV as a prism and each thruster as a cylinder marker with a thrust vector.
- Uses per-thruster position + angled thrust directions (M1..M8) to compute net force and torque.
- Lets you verify translational/rotational response in a basic 3D scene.

Run with:

```bash
python3 rov_simulator.py
```

The thruster geometry is configured in the `THRUSTERS` table inside `rov_simulator.py` (currently set to one inward-angled thruster per prism corner) so you can quickly tune positions/angles to match hardware revisions.
