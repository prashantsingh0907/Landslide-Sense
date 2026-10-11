"""
Landslide Risk Assessment Engine
Provides transparent heuristic risk scoring (Mode 2: Demonstration Mode)
and unified input validation for physical variables.
"""

from typing import Dict, Any, List, Optional, Tuple


def validate_environmental_inputs(
    rainfall_24h: Optional[float],
    rainfall_3d: Optional[float],
    rainfall_7d: Optional[float],
    slope: Optional[float],
    elevation: Optional[float],
    soil_moisture: Optional[float] = None,
    ndvi: Optional[float] = None,
    land_cover: Optional[str] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates physical environmental inputs against physically realistic bounds.
    Returns:
        (is_valid, validation_errors, unavailable_factors)
    """
    errors: List[str] = []
    unavailable: List[str] = []

    # Mandatory core parameters
    if rainfall_24h is None:
        errors.append("Rainfall (24h) is required.")
    elif rainfall_24h < 0.0 or rainfall_24h > 1500.0:
        errors.append(f"Rainfall (24h) out of realistic range [0, 1500 mm]: {rainfall_24h}")

    if rainfall_3d is None:
        errors.append("Rainfall (3d) is required.")
    elif rainfall_3d < 0.0 or rainfall_3d > 3500.0:
        errors.append(f"Rainfall (3d) out of realistic range [0, 3500 mm]: {rainfall_3d}")

    if rainfall_7d is None:
        errors.append("Rainfall (7d) is required.")
    elif rainfall_7d < 0.0 or rainfall_7d > 6000.0:
        errors.append(f"Rainfall (7d) out of realistic range [0, 6000 mm]: {rainfall_7d}")

    # Consistency checks across temporal rainfall aggregations
    if rainfall_24h is not None and rainfall_3d is not None and rainfall_24h > rainfall_3d:
        errors.append(f"Physical inconsistency: 24h rainfall ({rainfall_24h} mm) exceeds 3-day rainfall ({rainfall_3d} mm).")

    if rainfall_3d is not None and rainfall_7d is not None and rainfall_3d > rainfall_7d:
        errors.append(f"Physical inconsistency: 3-day rainfall ({rainfall_3d} mm) exceeds 7-day rainfall ({rainfall_7d} mm).")

    if slope is None:
        errors.append("Slope angle is required.")
    elif slope < 0.0 or slope > 90.0:
        errors.append(f"Slope angle out of physical range [0, 90 degrees]: {slope}")

    if elevation is None:
        errors.append("Elevation is required.")
    elif elevation < -500.0 or elevation > 9000.0:
        errors.append(f"Elevation out of terrestrial range [-500, 9000 m]: {elevation}")

    # Optional parameters with tracking of unavailable data
    if soil_moisture is None:
        unavailable.append("Soil Moisture (telemetry/in-situ sensor unavailable)")
    elif soil_moisture < 0.0 or soil_moisture > 100.0:
        errors.append(f"Soil moisture out of percentage bounds [0, 100%]: {soil_moisture}")

    if ndvi is None:
        unavailable.append("NDVI (satellite optical vegetation index unavailable)")
    elif ndvi < -1.0 or ndvi > 1.0:
        errors.append(f"NDVI out of theoretical bounds [-1.0, 1.0]: {ndvi}")

    if land_cover is None or str(land_cover).strip() == "" or str(land_cover).lower() == "none":
        unavailable.append("Land Cover Classification (thematic mapping unavailable)")

    is_valid = len(errors) == 0
    return is_valid, errors, unavailable


def calculate_heuristic_risk(
    rainfall_24h: float,
    rainfall_3d: float,
    rainfall_7d: float,
    slope: float,
    elevation: float,
    soil_moisture: Optional[float] = None,
    ndvi: Optional[float] = None,
    land_cover: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transparent, documented heuristic risk index for Demonstration Mode.
    Combines:
    1. Rainfall Trigger Index (0-40 pts):
       - 24h short-term intensity (max 18 pts)
       - 3d antecedent cumulative saturation (max 14 pts)
       - 7d prolonged catchment wetness (max 8 pts)
    2. Topographic Susceptibility Index (0-35 pts):
       - Slope angle criticality (max 27 pts, peak at 32-48°)
       - Elevation orographic / relief profile (max 8 pts)
    3. Surface & Land Cover Factor (0-25 pts):
       - Soil moisture saturation / pore-water potential (max 12 pts)
       - NDVI vegetative mechanical root reinforcement (max 7 pts)
       - Land cover vulnerability modifier (max 6 pts)

    Adjusts dynamically when optional variables are unavailable, explicitly
    annotating missing variables without silently assuming zero values.
    """
    valid, errors, unavailable = validate_environmental_inputs(
        rainfall_24h, rainfall_3d, rainfall_7d, slope, elevation,
        soil_moisture, ndvi, land_cover
    )
    if not valid:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    factor_details: List[Dict[str, Any]] = []

    # 1. RAINFALL TRIGGER SCORE (Nominal weight: 40 points)
    # 24h rainfall: 0 to 120+ mm
    r24_pts = min(18.0, (rainfall_24h / 120.0) * 18.0)
    factor_details.append({
        "factor": "24h Rainfall Intensity",
        "value": f"{rainfall_24h:.1f} mm",
        "score_contrib": round(r24_pts, 2),
        "max_possible": 18.0,
        "mechanism": "Short-duration intense infiltration generates shallow perched water tables and sudden loss of soil suction."
    })

    # 3d rainfall: 0 to 220+ mm
    r3d_pts = min(14.0, (rainfall_3d / 220.0) * 14.0)
    factor_details.append({
        "factor": "3-Day Cumulative Rainfall",
        "value": f"{rainfall_3d:.1f} mm",
        "score_contrib": round(r3d_pts, 2),
        "max_possible": 14.0,
        "mechanism": "Intermediate antecedent precipitation raises groundwater levels and increases regolith unit weight."
    })

    # 7d rainfall: 0 to 350+ mm
    r7d_pts = min(8.0, (rainfall_7d / 350.0) * 8.0)
    factor_details.append({
        "factor": "7-Day Cumulative Rainfall",
        "value": f"{rainfall_7d:.1f} mm",
        "score_contrib": round(r7d_pts, 2),
        "max_possible": 8.0,
        "mechanism": "Prolonged cumulative precipitation establishes deep hydrological connectivity across hillslope materials."
    })
    rainfall_subtotal = r24_pts + r3d_pts + r7d_pts

    # 2. TOPOGRAPHIC SUSCEPTIBILITY SCORE (Nominal weight: 35 points)
    # Slope: Peak susceptibility occurs typically between 30° and 48° for colluvial debris flows
    if slope < 10.0:
        slope_pts = 2.0
        slope_note = "Gentle slope (<10°); gravitational shear stress is substantially below shear strength."
    elif slope < 22.0:
        slope_pts = 6.0 + (slope - 10.0) / 12.0 * 6.0
        slope_note = "Moderate slope (10-22°); stable under dry conditions but prone to creep under high saturation."
    elif slope <= 46.0:
        # Critical slope zone
        slope_pts = 12.0 + (slope - 22.0) / 24.0 * 15.0
        slope_note = f"Critical landslide-prone gradient ({slope:.1f}°); driving shear forces approach internal friction angle."
    else:
        # Very steep slopes (>46°): Often bare bedrock or scoured cliffs, colluvial veneer is thinner
        slope_pts = max(16.0, 27.0 - (slope - 46.0) * 0.25)
        slope_note = f"Very steep to precipitous slope ({slope:.1f}°); high rockfall and rapid debris avalanche susceptibility."

    factor_details.append({
        "factor": "Slope Gradient",
        "value": f"{slope:.1f}°",
        "score_contrib": round(slope_pts, 2),
        "max_possible": 27.0,
        "mechanism": slope_note
    })

    # Elevation: High relief terrain experiences greater weathering, steeper valley profiles, and orographic precipitation
    elev_pts = min(8.0, max(1.0, (elevation / 3000.0) * 8.0))
    factor_details.append({
        "factor": "Terrain Elevation",
        "value": f"{elevation:.0f} m",
        "score_contrib": round(elev_pts, 2),
        "max_possible": 8.0,
        "mechanism": "Higher elevation belts experience enhanced orographic rainfall, periglacial weathering, and high relief energy."
    })
    topo_subtotal = slope_pts + elev_pts

    # 3. SURFACE & ENVIRONMENTAL CONDITIONING (Nominal weight: 25 points)
    surface_subtotal = 0.0
    surface_max_possible = 0.0

    # Soil moisture (max 12 pts if present)
    if soil_moisture is not None:
        surface_max_possible += 12.0
        sm_pts = min(12.0, (soil_moisture / 100.0) * 12.0)
        surface_subtotal += sm_pts
        factor_details.append({
            "factor": "Soil Moisture Saturation",
            "value": f"{soil_moisture:.1f}%",
            "score_contrib": round(sm_pts, 2),
            "max_possible": 12.0,
            "mechanism": "High volumetric soil water content increases positive pore-water pressure and degrades effective shear cohesion."
        })

    # NDVI (max 7 pts if present): higher NDVI means stronger root anchorage, lower hazard
    if ndvi is not None:
        surface_max_possible += 7.0
        # Invert: NDVI ~ 0.8 => low hazard (~1 pt); NDVI ~ 0.1 => high hazard (~6.5 pts)
        clamped_ndvi = max(-0.2, min(1.0, ndvi))
        ndvi_hazard_factor = max(0.0, min(1.0, (0.8 - clamped_ndvi) / 0.9))
        ndvi_pts = ndvi_hazard_factor * 7.0
        surface_subtotal += ndvi_pts
        factor_details.append({
            "factor": "Vegetation Density (NDVI)",
            "value": f"{ndvi:.2f}",
            "score_contrib": round(ndvi_pts, 2),
            "max_possible": 7.0,
            "mechanism": "Sparse root systems afford minimal mechanical reinforcement; dense canopy buffers rainfall impact."
        })

    # Land cover (max 6 pts if present)
    land_cover_weights = {
        "barren/sparse": 5.8,
        "settlement": 4.8,
        "agricultural": 3.4,
        "shrubland": 2.2,
        "forest": 0.8,
    }
    if land_cover is not None and str(land_cover).strip() != "":
        lc_key = str(land_cover).strip().lower()
        surface_max_possible += 6.0
        lc_pts = land_cover_weights.get(lc_key, 3.0)
        surface_subtotal += lc_pts
        factor_details.append({
            "factor": "Land Cover Classification",
            "value": str(land_cover).capitalize(),
            "score_contrib": round(lc_pts, 2),
            "max_possible": 6.0,
            "mechanism": f"Class '{land_cover}' influences slope runoff coefficient, terracing integrity, and anthropogenic cut-slope disturbance."
        })

    # Total score normalization
    # Total available baseline = 40 (rain) + 35 (topo) = 75 points
    # Plus whatever surface parameters were provided (0 to 25 points)
    total_raw_score = rainfall_subtotal + topo_subtotal + surface_subtotal
    total_scale_max = 75.0 + surface_max_possible

    # Normalize to [0, 100] standard index
    normalized_score = min(100.0, max(0.0, (total_raw_score / total_scale_max) * 100.0))

    # Risk level categorization
    if normalized_score < 35.0:
        risk_level = "Low"
        alert_color = "#22c55e"  # Green
        precautionary_message = (
            "BASELINE STABILITY: Current hydrological and slope parameters indicate low immediate "
            "landslide susceptibility. Routine seasonal vigilance and normal slope maintenance are recommended."
        )
    elif normalized_score < 70.0:
        risk_level = "Medium"
        alert_color = "#f59e0b"  # Amber
        precautionary_message = (
            "ELEVATED RISK DETECTED: Significant antecedent rainfall or critical slope gradients detected. "
            "Inspect hillslope drainage channels, monitor cut-slopes along roads, and exercise caution in areas "
            "with past instability or tension cracks."
        )
    else:
        risk_level = "High"
        alert_color = "#ef4444"  # Red
        precautionary_message = (
            "CRITICAL HAZARD WARNING: Combination of high rainfall accumulation, steep topography, and high saturation "
            "presents severe risk of slope failure or debris flow. Avoid unstable slopes and stream valleys, inform local "
            "disaster management authorities, and watch for sudden muddy seepage or soil movement."
        )

    return {
        "mode": "Demonstration (Configurable Transparent Heuristic)",
        "risk_score": round(normalized_score, 1),
        "risk_level": risk_level,
        "alert_color": alert_color,
        "precautionary_message": precautionary_message,
        "factor_breakdown": factor_details,
        "unavailable_factors": unavailable,
        "rainfall_subtotal": round(rainfall_subtotal, 1),
        "topo_subtotal": round(topo_subtotal, 1),
        "surface_subtotal": round(surface_subtotal, 1) if surface_max_possible > 0 else None,
        "total_scale_max": total_scale_max,
        "disclaimer": (
            "Illustrative heuristic risk index for demonstration purposes only. "
            "Not a trained ML probability or official civil warning."
        ),
    }
