"""WGS84 input uses longitude,latitude; all geometric features use metres."""
from dataclasses import dataclass
import numpy as np
from pyproj import CRS, Transformer


@dataclass
class LocalProjection:
    epsg: int
    forward: Transformer
    inverse: Transformer

    @classmethod
    def from_coordinates(cls, longitude, latitude):
        lon, lat = float(np.median(longitude)), float(np.median(latitude))
        if not -80 <= lat <= 84:
            raise ValueError("UTM projection requires latitude between -80 and 84")
        zone = min(60, max(1, int((lon + 180) // 6) + 1))
        epsg = (32600 if lat >= 0 else 32700) + zone
        return cls(epsg, Transformer.from_crs(4326, CRS.from_epsg(epsg), always_xy=True),
                   Transformer.from_crs(epsg, 4326, always_xy=True))

    def project(self, longitude, latitude):
        x, y = self.forward.transform(longitude, latitude)
        return np.column_stack([x, y])


def bearing_degrees(dx, dy):
    """Grid bearing clockwise from north, consistently used for GPS and roads."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def angular_difference(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
