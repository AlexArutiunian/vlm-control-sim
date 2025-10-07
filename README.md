# ANY_ROBOT

## How to launch
```bash
uv run any_robot.py --robot g1
```

[g1, h1_2] - avaible options

## How does it works?

VLM получает  промпт с полным описанием робота, значениями и наименованиями текущих суставов и командой которую задал пользователь на естественном языке.

VLM возвращает json массив с джоинтами и значениями углов на которые их установить. 

Например: по команде "Подними правую руку вверх" ожидается что-то вроде:

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

![alt text](recorded/h1_2.gif)
![alt text](recorded/h1_2-2.gif)
![alt text](recorded/g1.gif)