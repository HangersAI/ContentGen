import os
import time
from modules.tagging import TaggingEngine


# Take about 10 minute download weight for the first run, require > 14Gb VRAM
modelEngine = TaggingEngine(model_path= 'Qwen/Qwen-VL-Chat-Int4')

image_path = "examples/cong_nghe/10.jpg"


prediction = modelEngine.general_tagging(image_path=image_path)


print(prediction)


## Example output for this run
"""{'message': 'Success',
 'status_code': 200,
 'infos': {'category': 'Smartwatch',
  'color': 'Pink',
  'material': 'Plastic',
  'power_source': 'Battery'}}"""