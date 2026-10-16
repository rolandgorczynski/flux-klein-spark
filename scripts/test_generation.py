"""Run against the local Gradio service after model access has been granted."""
import argparse
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
from gradio_client import Client, handle_file

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:7861')
parser.add_argument('--size', type=int, default=512)
parser.add_argument('--lora', action='append', default=[])
parser.add_argument('--name', default='spark-generation-test')
args = parser.parse_args()
out = Path('/data/output')
base = out / 'spark-test-input.png'
if not base.exists():
    image = Image.new('RGB', (512, 512), '#e8dfce')
    draw = ImageDraw.Draw(image)
    draw.ellipse((150, 120, 360, 380), fill='#42758c')
    draw.rectangle((170, 110, 340, 170), fill='#42758c')
    image.save(base)
client = Client(args.url, verbose=False)
start = time.monotonic()
result = client.predict(
    handle_file(str(base)), [],
    'Turn the blue object into a beautiful blue ceramic vase with a small bunch of yellow flowers, product photography, soft studio lighting, beige background',
    '', '', args.lora, 42, False, 1.0, 4, 'None', 'Custom', args.size, args.size,
    'Stretch', '#000000', 1, 'Sequential seed (+1 each)', 0.4, 1.4,
    *([1.0] * 6), api_name='/infer',
)
report = {'seconds': round(time.monotonic() - start, 2), 'size': args.size,
          'loras': args.lora, 'result': result}
(out / f'{args.name}.json').write_text(json.dumps(report, default=str, indent=2))
print(json.dumps(report, default=str, indent=2), flush=True)
