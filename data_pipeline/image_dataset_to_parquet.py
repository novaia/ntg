import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image
import os
import io
import json

samples_per_file = 10_000

root_dir = 'data/datasets/world-heightmaps-256-v1'
df = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
df = df.sample(frac=1).reset_index(drop=True)

print(df.head())

def save_table(image_data, table_number):
    print(f'Entries in table {table_number}: {len(image_data)}')
    schema = pa.schema(
        fields=[
            ('heightmap', pa.struct([('bytes', pa.binary()), ('path', pa.string())])),
            ('latitude', pa.string()),
            ('longitude', pa.string())
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'heightmap': {'_type': 'Image'},
                        'latitude': {'_type': 'Value', 'dtype': 'string'},
                        'longitude': {'_type': 'Value', 'dtype': 'string'}
                    }
                }
            }).encode('utf-8')
        }
    )

    table = pa.Table.from_pylist(image_data, schema=schema)
    pq.write_table(table, f'data/world-heightmaps-256-parquet/{str(table_number).zfill(4)}.parquet')

image_data = []
samples_in_current_file = 0
current_file_number = 0
for i, row in df.iterrows():
    if samples_in_current_file >= samples_per_file:
        save_table(image_data, current_file_number)
        image_data = []
        samples_in_current_file = 0
        current_file_number += 1
    samples_in_current_file += 1
    image_path = row['file_name']
    with Image.open(os.path.join(root_dir, image_path)) as image:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_dict = {
            'heightmap': {
                'bytes': image_bytes.getvalue(),
                'path': image_path
            },
            'latitude': str(row['latitude']),
            'longitude': str(row['longitude'])
        }
        image_data.append(image_dict)

save_table(image_data, current_file_number)
