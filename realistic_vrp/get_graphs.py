# Importing required libraries

import torch

import overpy
import requests 
from bs4 import BeautifulSoup
from urllib import parse
import random
import pickle
import numpy as np
import pandas as pd

#setup OSRM 
from routingpy import OSRM
from routingpy.routers import options
options.default_timeout=None



class OSM_graph():
    """
    For a given address query [city_name, shop_type], first we get co-ordinates and distance values from Open Street Map
    Then generate a dataset of n_instances having graph structure information (node and edge features)

    Other shop types : supermarket, convenience, clothes, hairdresser, car_repair, bakery
    """

    def get_area_code(self, address):
        """
        Gets the area code
        """
        url = "https://www.openstreetmap.org/geocoder/search_osm_nominatim?query=" + parse.quote(address)
        r = requests.get(url) 
        soup = BeautifulSoup(r.content, 'html5lib')
        osm_link = soup.find('a', attrs = {'class':'set_position'})
        relation_id = osm_link.get('data-id').strip()	
        return int(relation_id) + 3600000000 # 3600000000 is the offset in ids 



    def get_coordinates(self, address, shop_type):
        # Setting up 
        area_code = self.get_area_code(address)
        api = overpy.Overpass()

        request = api.query(f"""area({area_code});
        (node[shop={shop_type}](area);
        way[shop={shop_type}](area);
        rel[shop={shop_type}](area);
        ); out center;""")

        coords = [[float(node.lon), float(node.lat)] for node in request.nodes]
        coords += [[float(way.center_lon), float(way.center_lat)] for way in request.ways]
        coords += [[float(rel.center_lon), float(rel.center_lat)] for rel in request.relations]

        print(f"Total {len(coords)} points found on the map for search query: {address, shop_type}")
        return coords



    def generate_graphs(self, address, shop_type, graph_size, n_instances, edge_type = "distance", out_path = "./cvrp_data/test.pk"):
        """
        Gets the coordinates for the given query and generates node_features and edge_features
        """
        coordinates = self.get_coordinates(address, shop_type)

        client = OSRM(base_url="https://router.project-osrm.org")
        instances = []

        for _ in range (n_instances):
            # Node coordinates for the graph are obtained by randomly sampling graph_size nodes from all queried locations on OSM
            instance_coords = random.sample(coordinates, graph_size)
            dist_matrix = client.matrix(locations=instance_coords, profile="car")

            if edge_type == "distance":
                edge_features = np.array(dist_matrix.distances)
            elif edge_type == "time":
                edge_features = np.array(dist_matrix.durations)

            edge_features = torch.from_numpy(edge_features).float()
            instance = {"coordinates": instance_coords, "edge_features": edge_features}
            instances.append(instance)


        # Now generating node features for all instances -- assigning first node as depot with no demand, 
        # and all others with equal demands (for now)
        # Node features: [is_depot, is_customer, demand, coordinate_1, coordinate_2]
        
        all_graphs = []
        demand = 1/(len(instances[0]["coordinates"])-1) #Equal demand for all customers
        
        for instance in instances:
            graph_nodes = []
            coordinates = instance["coordinates"]

            for idx, node_coordinates in enumerate(coordinates):

                if idx == 0:
                    node_features = np.append(np.array([1, 0, 0]), node_coordinates)
                    graph_nodes = node_features
                else:
                    node_features = np.append(np.array([0, 1, demand]), node_coordinates)
                    graph_nodes = np.vstack((graph_nodes, node_features))

            graph_nodes = torch.from_numpy(graph_nodes).float()
            graph_struct = {"node_features": graph_nodes, "edge_features": instance["edge_features"], "coordinates": coordinates}
            all_graphs.append(graph_struct)

        return all_graphs