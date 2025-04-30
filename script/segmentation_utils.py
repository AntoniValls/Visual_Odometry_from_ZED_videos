import osmnx as ox
import pandas as pd
import geopandas as gpd


def street_segmentation(initial_point, zone, area=750): # NOT FULLY UNDERSTOOD
    '''
    Function that checks if the point is on the graph and the distance to the closest edge of the graph

    Parameters:
        initial_point: (1,2) array of the coordinates (latitude, longitude) of the initial point of the sequence
        zone: UTM zone of the region  e.g. ("+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs") for KITTI sequence in Germany
        area: The radius of the area around the initial point that information is extracted.
    Returns:
        edges: Graph containing the center of the streets of an area around the initial point.
        road_area: Geoseries containg the area of the streets dimensioned to the number of lanes, if this is available
        walkable_area_gdf: Geoseries containg the walkable area (buildings minus streets)
    '''
    
    G = ox.graph_from_point((initial_point[0], initial_point[1]), dist=area, network_type='drive')
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # Check how to pass this info to the function.
    edges = edges.to_crs(zone)

    # Extract lane info, coerce -> invalid parsing will be set as NaN
    edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce')

    # Assumes that if info is NaN there is 1 lane
    edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce').fillna(1)

    # Creates the road space as a function of the number of lanes
    edges['buffer_size'] = edges['lanes'] * 3.5  # Assuming each lane is ~3.5 meters wide

    road_buffer = edges.geometry.buffer(edges['buffer_size'])
    road_area = gpd.GeoSeries(road_buffer, crs=edges.crs)

    buildings = ox.features_from_point(initial_point, tags={"building": True}, dist=area)
    buildings = buildings.to_crs(zone)
    building_area = gpd.GeoSeries(buildings.unary_union,crs=edges.crs)

    convex_hull = buildings.unary_union.convex_hull
    total_area = gpd.GeoSeries([convex_hull], crs=buildings.crs)

    # Identify intersections (crossings) in the graph using OpenStreetMap's crossing tag
    crossings = ox.features_from_point(initial_point, tags={"highway": "crossing"}, dist=area)
    crossings = crossings.to_crs(zone)

    # Buffer around crossings to represent their area
    # Match crossings to the nearest road segment and assign buffer size based on road width
    crossings = gpd.sjoin(crossings, edges[['geometry', 'buffer_size']], how='left', predicate='intersects')
    crossings['buffer_size'] = crossings['buffer_size'].fillna(5)  # Default to 5 meters if no match is found
    crossings_buffer = crossings.geometry.buffer(crossings['buffer_size']*1.4)
    crossings_area = gpd.GeoSeries(crossings_buffer.unary_union, crs=edges.crs)
    
    # Subtract the building footprints from the buffered street area to get walkable space

    walkable_area = total_area.difference(building_area)
    unified_road = gpd.GeoSeries(road_buffer.unary_union, crs=edges.crs)
    walkable_area = walkable_area.difference(unified_road)

    # Convert to GeoDataFrame for easy plotting
    walkable_area_gdf = gpd.GeoDataFrame(geometry=walkable_area, crs=edges.crs)

    return edges,road_area,walkable_area_gdf,building_area,crossings_area