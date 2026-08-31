from dataclasses import dataclass


@dataclass
class DetectionConfig:
    eps: float = 0.8
    min_samples: int = 25
    min_height_m: float = 3.0
    max_pole_radius_m: float = 1.6
    crossarm_top_band_m: float = 1.0
    ground_grid_m: float = 1.0
    min_points_per_cluster: int = 80
    support_radius_factor: float = 1.8
    base_percentile: float = 1.0
    top_percentile: float = 99.8
    keep_only_tallest_per_file: bool = True
    shaft_radius_m: float = 0.45
    top_cap_exclude_m: float = 0.15
    crossarm_search_drop_m: float = 4.0
    crossarm_band_thickness_m: float = 0.6
    crossarm_min_radius_m: float = 0.35
    crossarm_max_radius_m: float = 2.2
    crossarm_min_drop_from_top_m: float = 0.2
    crossarm_max_drop_from_top_m: float = 2.0
    crossarm_min_length_m: float = 0.5
    prefer_pole_height_m: float = 14.0
    max_reasonable_pole_height_m: float = 18.0
    bulk_min_point_count: int = 70
    bulk_min_height_m: float = 6.0
    bulk_max_height_m: float = 32.0
    bulk_nms_radius_m: float = 2.4
