"""EDA-only Folium layers. Reference routes are never an inference UI response."""
from pathlib import Path
import folium
import numpy as np


def raw_trip_map(trip, destination, max_road_segments=15000):
    track = trip.track
    center = [float(track.latitude.median()), float(track.longitude.median())]
    result = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap", prefer_canvas=True)
    roads = folium.FeatureGroup(name="Road network (display subset)", show=True)
    lonlat = trip.nodes[["longitude", "latitude"]].to_numpy()
    endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int)
    segments = lonlat[endpoints]
    lo = track[["longitude", "latitude"]].min().to_numpy() - .002
    hi = track[["longitude", "latitude"]].max().to_numpy() + .002
    visible = np.flatnonzero(((segments.max(axis=1) >= lo) & (segments.min(axis=1) <= hi)).all(axis=1))
    # Display thinning only; processing always uses every arc.
    if len(visible) > max_road_segments:
        visible = visible[np.linspace(0, len(visible)-1, max_road_segments, dtype=int)]
    geojson = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"arc_id": int(i)},
         "geometry": {"type": "LineString", "coordinates": segments[i].tolist()}} for i in visible]}
    folium.GeoJson(geojson, style_function=lambda _: {"color": "#8998a5", "weight": 1, "opacity": .5},
                   tooltip=folium.GeoJsonTooltip(fields=["arc_id"])).add_to(roads)
    roads.add_to(result)
    gps = folium.FeatureGroup(name="GPS observations (blue)")
    # All points are retained; canvas rendering avoids thousands of DOM icons.
    for i, row in track.iterrows():
        folium.CircleMarker([row.latitude, row.longitude], radius=2, color="#176fc1",
                            fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"Observation {i}; t={row.timestamp_s:g}s").add_to(gps)
    gps.add_to(result)
    if trip.route is not None:
        reference = folium.FeatureGroup(name="Supplied reference route (green)")
        coordinates = [segments[int(a)][:, ::-1].tolist() for a in trip.route.arc_id]
        folium.PolyLine(coordinates, color="#23955b", weight=4, opacity=.8).add_to(reference)
        reference.add_to(result)
    folium.LayerControl(collapsed=False).add_to(result)
    result.fit_bounds([[float(track.latitude.min()), float(track.longitude.min())],
                       [float(track.latitude.max()), float(track.longitude.max())]])
    legend = '<div style="position:fixed;bottom:25px;left:25px;background:white;padding:12px;z-index:9999;border:1px solid #999;font:13px sans-serif">'
    legend += '<b>PathSnap EDA</b><br><span style="color:#176fc1">●</span> GPS observations<br><span style="color:#23955b">━</span> Supplied reference<br><span style="color:#8998a5">━</span> Road network<br>Reference layers are for development EDA only.</div>'
    result.get_root().html.add_child(folium.Element(legend))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    result.save(str(destination))
    return result
