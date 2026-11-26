import json
import os

with open('frontend/public/india_states.geojson', 'r', encoding='utf-8') as f:
    geojson = json.load(f)

lat_min, lat_max = 6.5, 37.5
lon_min, lon_max = 65.5, 100.5

def to_xy(lon, lat, width=720, height=800):
    x = (lon - lon_min) / (lon_max - lon_min) * width
    y = (lat_max - lat) / (lat_max - lat_min) * height
    return round(x, 2), round(y, 2)

state_paths = []
all_outlines = []

for feat in geojson['features']:
    geom = feat['geometry']
    state_name = feat.get('properties', {}).get('NAME_1') or feat.get('properties', {}).get('ST_NM') or feat.get('properties', {}).get('name') or 'State'
    gtype = geom['type']
    polys = geom['coordinates'] if gtype == 'MultiPolygon' else [geom['coordinates']]
    
    combined_d = []
    for poly in polys:
        for ring in poly:
            points = [to_xy(pt[0], pt[1]) for pt in ring]
            if len(points) < 3:
                continue
            d_parts = [f"M {points[0][0]} {points[0][1]}"]
            for pt in points[1:]:
                d_parts.append(f"L {pt[0]} {pt[1]}")
            d_parts.append("Z")
            combined_d.append(" ".join(d_parts))
    
    if combined_d:
        d_str = " ".join(combined_d)
        state_paths.append(f'    <path class="state-boundary" d="{d_str}" data-name="{state_name}" />')

svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 800" class="india-vector-map">
  <defs>
    <style>
      .state-boundary {{
        fill: none;
        stroke: rgba(51, 65, 85, 0.45);
        stroke-width: 0.9;
        vector-effect: non-scaling-stroke;
        pointer-events: none;
      }}
      .india-outline {{
        fill: none;
        stroke: #334155;
        stroke-width: 1.5;
        vector-effect: non-scaling-stroke;
        pointer-events: none;
      }}
    </style>
  </defs>
  <g class="states-layer">
{os.linesep.join(state_paths)}
  </g>
</svg>'''

os.makedirs('frontend/public/maps', exist_ok=True)
with open('frontend/public/maps/india.svg', 'w', encoding='utf-8') as out:
    out.write(svg_content)

print(f"Saved frontend/public/maps/india.svg with {len(state_paths)} states")
