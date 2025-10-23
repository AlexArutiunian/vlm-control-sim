# ANY_ROBOT



## How to launch

```bash
pip install uv
```

```bash
uv run any_robot.py --robot h1_2
```
[g1, h1_2, h1] - avaible options (better if h1_2 or g1)

- Voice control 2 robots: 
Key V to launch the dialog with realtime llm
```bash
uv run any_ag_micro.py
```

- 3 twin-robots: 

```bash
uv run any_3h1_2.py
```

- The best version with avoiding obstacles by A*: 

```bash
uv run any_robots_map.py
```



## How does it works?

VLM receives a prompt with a full description of the robot, the values and names of the current joints, and a command given by the user in natural language.

VLM returns a json array with the joints and the angle values to set them to. 

For example: the command "Raise your right hand up" is expected to do something like:

```json
[
{
"frame": [
{"name": "right_shoulder_pitch_joint", "angle": -185},
{"name": "right_elbow_joint", "angle": 90}
],
"duration": 1
}
]
```

## Examples

![alt text](recorded/h1_2_hold_white.gif)
![alt text](recorded/g1_replace_sphere.gif)
![alt text](recorded/h1_2_hold_white_2nd_view.gif)
![alt text](recorded/red_box_wide.gif)
<p align="center">
  <img src="recorded/hello3x.gif" alt="alt text">
</p>