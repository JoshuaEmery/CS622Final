import streamlit as st
import json
import pandas as pd
import plotly
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import scipy as sp
import streamlit as st
import csv
import networkx as nx

def main():
    # Set page title and favicon
    st.set_page_config(page_title='Travel Plan: Finding the Shortest Path', page_icon=':memo:')

    # Create sidebar menu
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Go to', ('Home', 'Abstract',
     'Introduction to Graphs', 
     'Graph Components',
     'Types of Graphs',
     'Shortest Path Algorithm',
     "Djikstra's Algorithm",
     'Sample Data Set',
     'Visualize Data Set', 
     'Conclusion'))

    # Display corresponding page based on menu selection
    if page == 'Home':
        display_home()
    elif page == 'Abstract':
        display_abstract()
    elif page == 'Introduction to Graphs':
        display_intro()
    elif page == 'Types of Graphs':
        display_types_of_graphs();
    elif page == 'Graph Components':
        graph_components()
    elif page == "Djikstra's Algorithm":
        djikstras()
    elif page == 'Shortest Path Algorithm':
        shortestpath()
    elif page == 'Sample Data Set':
        sample_data()
    elif page == 'Visualize Data Set':
        visualization()
    elif page == 'Results':
        display_results()
    else:
        display_conclusion()

def display_home():
    st.title('Travel Plan: Finding the Shortest Path ')
    st.subheader('Authors')
    st.markdown("""
                * Umutbek Abdimanan uulu  - [abdimananuuluumutbe@cityuniversity.edu](mailto:abdimananuuluumutbe@cityuniversity.edu)
                * Ajay Shrikrishna Naik - [naikajayshrikrishna@cityuniversity.edu](mailto:naikajayshrikrishna@cityuniversity.edu)
                * Joshua Emery - [emeryjoshua@cityuniversity.edu](mailto:emeryjoshua@cityuniversity.edu)
                * Dinesh Chakravarthy Perni - [pernidinesh@cityuniversity.edu](mailto:pernidinesh@cityuniversity.edu)
                """)

def display_abstract():
    st.title('Abstract:')
    st.write("""Traveling to different destinations is a major part of our lives. People constantly move from one location to another. In recent years we have begun to use navigation devices to guide our travel. How do these devices calculate a path? How do we make decisions about our movement? If you were trying to move from your bedroom to the kitchen, how would you go about this process? It would be inefficient to walk from your bedroom, then to the laundry room, then outside to your backyard, and finally to your kitchen. It would be much faster to walk directly from your bedroom to your kitchen. However, your bedroom might not directly connect to your kitchen. Some traversals through other locations may be required, but which locations, and in what order? In a trip with locations that are close to each other, there are a limited number of paths which connect the locations. It may be feasible to test each different path to find the shortest one. However, when examining locations that are further apart the number of unique paths from location a to location b grows exponentially. It becomes impractical to physically traverse each path to discover the most efficient route. To solve this problem, we have decided to research the Graph data structure. The breadth-first search and Dijkstra's shortest path algorithm with be tested and applied. Through these methods, we hope to deepen our knowledge about computation, algorithms, and data structures while creating a project related to this subject.  

City and location data will be used in a graph data structure to find the shortest path using Breadth-first and the Dijkstra algorithm. Performance and time complexities will be tested and recorded with the goal of finding the most efficient examples of these algorithms.  

Two algorithms (Breadth-first search, and Dijkstra’s shortest path algorithm) and Graph data structure will be used. """)

def display_intro():
    st.title('Introduction to Graphs:')
    st.write("""
    A graph is a structure that comprises a set of vertices and a set of edges. So, in order to have a graph we need to define the elements of two sets: vertices and edges. The graph is nothing but an organized representation of data. It helps us to understand the data. Data are the numerical information collected through observation. The word data came from the Latin word Datum which means “something given”. An edge, if it exists, is a link or a connection between any two vertices of a graph, including a connection of a vertex to itself. The idea behind edges is that they indicate, if they are present, the existence of a relationship between two objects, that we imagine the edges to connect. 
    """)
    st.write("""
    We usually indicate with V = {v1, v2 , …vn} the set of vertices, and E= {e1,e2,..em} set of edges. We can then define a graph Shape G as the structure G = (V, E) which models the relationship between the two sets: 
    """)
    st.image('fig1.png', caption='Fig-1 Graph')
    st.subheader('The role of graphs in solving navigation problems:')
    st.write("""
    Graphs are essential for solving navigation problems because they offer a convenient way to represent and analyze intricate networks of interrelated routes, such as airline or road networks. Using graphs to represent such networks, we can use established algorithms to find the quickest path between two points, pinpoint significant nodes and routes, and optimize routes and schedules. we represent the network as a graph where each point corresponds to a location, and each line represents a route between locations. Then, we can apply graph algorithms to compute the shortest path between any two points in the network.
    """)

    st.subheader('Limitations and convenience of using graphs for navigation:')
    st.write("""
   Using graphs, we can easily find the shortest path and neighbors of the nodes. Graphs can handle large amounts of data and can easily be distributed across multiple machines. Because of its non-linear structure, helps in understanding complex problems and their visualization. Limitations are few Some graph algorithms have high time complexity, which can slow down the performance of a system. It can have large memory complexity.
    """)


def graph_components():
    st.title('Graph Components')
    st.subheader('Vertices:')

    st.write("""
    Vertices are the fundamental units of the graph. Sometimes, vertices are also known as vertices or nodes. Every node/vertex can be labeled or unlabeled.
    """)

    st.subheader('Edges:')

    st.write("""
Edges are drawn or used to connect two nodes of the graph. It can be an ordered pair of nodes in a directed graph. Edges can connect any two nodes in any possible way. There are no rules. Sometimes, edges are also known as arcs. Every edge can be labeled/unlabeled.
    """)

    st.write("""
    Graphs are used to solve many real-life problems. Graphs are used to represent networks. The networks may include paths in a city or telephone network or circuit network. Graphs are also used in social networks like LinkedIn, and Facebook. For example, each person is represented on Facebook with a vertex (or node). Each node is a structure and contains information like person id, name, gender, locale, etc ( De Luca, G. (2022, November 18). Introduction to Graph Theory | Baeldung on Computer Science. Baeldung on Computer Science).
    """)

def display_types_of_graphs():
    st.title('Types of Graphs')
    st.subheader('Directed Graph:')
    st.write("""
    In directed graphs, the edges direct the path that must be taken to travel between connected nodes. The edges are typically represented as arrows.
    """)
    st.image('fig2.png', caption='Fig-2 Directed Graph')
    st.subheader('Undirected Graph:')
    st.write("""
Undirected graphs do not show the direction which must be taken between nodes. Instead, travel between nodes is allowed along an edge in either direction. There are no loops or multiple edges in undirected graphs.
    """)
    st.image('fig3.png', caption='Fig-3 Undirected Graph')
    st.subheader('Weighted Graph:')
    st.write("""
A weighted graph is a graph with weighted edges. The weights may represent factors like cost or the distance required to travel between nodes.
    """)
    st.image('fig5.png', caption='Fig-5 Simple Graph')
    st.subheader('Simple Graph:')
    st.write("""
A simple graph can also be referred to as a strict graph. Simple graphs are undirected, and the edges are not weighted; these may or may not have any connected edges. They also have no loops and lack multiple edges. Simple graphs consist of nodes and vertices. A node is a vertex which is the 'dot' on the graph. Edges connect each node and may have a weight associated with them and flow in specific directions between nodes.
    """)
    st.image('fig2.png', caption='Fig-2 Directed Graph')

def djikstras():
    st.title("Djikstra's Algorithm")
    st.write("""
    Dijkstra's algorithm (/ˈdaɪkstrəz/ DYKE-strəz) is an algorithm for finding the shortest paths between nodes in a weighted graph, which may represent, for example, road networks
The algorithm exists in many variants. Dijkstra's original algorithm found the shortest path between two given nodes, but a more common variant fixes a single node as the "source" node and finds the shortest paths from the source to all other nodes in the graph, producing the shortest path.
For a given source node in the graph, the algorithm finds the shortest path between that node and every other. It can also be used for finding the shortest paths from a single node to a single destination node by stopping the algorithm once the shortest path to the destination node has been determined. For example, if the nodes of the graph represent cities and costs of edge paths represent driving distances between pairs of cities connected by a direct road (for simplicity, ignore red lights, stop signs, toll roads, and other obstructions), then Dijkstra's algorithm can be used to find the shortest route between one city and all other cities. 
Suppose you would like to find the shortest path between two intersections on a city map: a starting point and a destination. Dijkstra's algorithm initially marks the distance (from the starting point) to every other intersection on the map with infinity  ( Wikipedia contributors. (2023c, March 3). Dijkstra’s algorithm.)
    """)
    st.subheader("Applications of Djikstra's Algorithm")
    st.write("""
    Further, with the discussion, it has various real-world use cases, some of the applications are the following:
    """)
    st.markdown("""
    1.	For map applications, it is hugely deployed in measuring the least possible distance and check direction amidst two geographical regions like Google Maps, discovering map locations pointing to the vertices of a graph, calculating traffic and delay-timing, etc.
    2.	For telephone networks, this is also extensively implemented in the conducting of data in networking and telecommunication domains for decreasing the obstacle taken place for transmission.
    3.	Wherever addressing the need for shortest path explications either in the domain of robotics, transport, embedded systems, laboratory or production plants, etc, this algorithm is applied.

    """)
    st.write("""
    Besides that, other applications are road conditions, road closures and construction, and IP routing to detect Open Shortest Path First (Tyagi, n.d.)
    """)

def shortestpath():
    st.title('Shortest Path Algorithms')
    st.write("""
    For a given graph, the shortest path algorithms determine the minimum cost of the path from the source vertex to every vertex in a graph.
The path is the movement traced across a sequence of vertices V1, V2, V3,.., VN in a graph. The cost of the path is the sum of the cost associated with all the edges in a graph. This is represented as:
    """)
    st.image('fig6.jpg')
    st.subheader('Where does the shortest path algorithm find its applications?')
    st.markdown("""
    *	We can use the shortest path algorithm to find the cheapest way to send the information from one computer to another within a network.

    *	We can use the shortest path algorithm to find the best route between the two airports.

    *	We can use it to find a road route that requires a minimum distance from a starting source to an ending destination. We can determine the route which requires minimum time to an ending destination

    """)
    st.subheader('Shortest Path Problems:')
    st.write("""
    The single source shortest path problem is used to find the minimum cost from a single source to all other vertices.
    """)
    st.write("""
The all-pairs shortest path problem is used to find the shortest path between all pairs of vertices. The all-pair shortest path algorithm also known as the Floyd-Warshall algorithm is used to find the all-pair shortest problem from a given weighted graph. (K. (n.d.). Shortest Path Algorithms. Krivalar.com).
    """)
    st.write("""
Moving forward we are going to find the shortest distance between random cities using the single source’s shortest path so, to solve this we use Dijkstra's algorithm.
    """)
    st.write("""
As part of the project we work to find the shortest path between two cities in King county and we use Dijkstra's algorithm and BFS (Breadth First to find the shortest path between two areas and compare the performance of algorithms using them.
    """)

def sample_data():
    st.title('Sample Data and Code Snippets')
    st.write("""
    To test the model, we have used the Test.csv dataset which is holding distances between various cities. We use this data to test our algorithms for finding the shortest distance also the performance of algorithms is calculated. Sample data we are using as follows:
    """)
    st.image('fig6.png', caption="""Fig-6 Data in Test.csv used to find Shortest Distance.
""")
    st.subheader('Code Examples')
    st.write("""
    To implement the Dijkstra’s algorithm, we are reading the csv file from the Main.py file and removing the first line of csv as it is holding the header line for our data. Whole execution starts from the Main.py file where we call different algorithms to execute.
    """)
    st.subheader('Code Snippet 1 Main.py')
    st.image('img7.png')
    st.subheader('Code Snippet 2')
    st.image('img8.png')
    st.subheader('Code Snippet 3')
    st.image('img9.png')
    st.write("""
    The Dijkstra Algorithm class takes in a dictionary of routes as input and initializes an instance variable self.routes with this dictionary. The Dijkstra method takes in the start city and end city as input and initializes three empty dictionaries - shortest distance, path, and predecessor.
    """)
    st.subheader('Code Snippet 4')
    st.image('img10.png')
    st.write("""
    This loop finds the node with the smallest value of the shortest distance that has not yet been visited. This node becomes the new minNode.

This loop updates the shortest distance and predecessor dictionaries for each adjacent node of the minNode. If the distance from the minNode to a child node plus the distance to the minNode is less than the current shortest distance for that child node, then the shortest distance and predecessor dictionaries are updated. The minNode is then removed from the self.routes dictionary.

This loop backtracks from the end_city to the start_city, adding each node to the path list. If a node is reached that does not have a predecessor in the predecessor dictionary, then the path is not reachable and the loop is broken.

    """)




             

def visualization():
    st.title('Graph of King County Cities')
    with open('test.csv', 'r') as file:
    #Create the network graph using networkx
        if file is not None:     
            df=pd.read_csv(file)
            #Read the Cities data from the cps
            A = list(df["StartCity"].unique())
            B = list(df["EndCity"].unique())
            node_list = set(A+B)
            G = nx.Graph() #Use the Graph API to create an empty network graph object
            
            #Add nodes and edges to the graph object
            for i in node_list:
                G.add_node(i)
            for i,j in df.iterrows():
                G.add_edges_from([(j["StartCity"],j["EndCity"])])
            pos = nx.spring_layout(G, k=0.5, iterations=100)
            for n, p in pos.items():
                G.nodes[n]['pos'] = p
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=1,color='#888'),
                hoverinfo='none',
                mode='lines')

            for edge in G.edges():
                x0, y0 = G.nodes[edge[0]]['pos']
                x1, y1 = G.nodes[edge[1]]['pos']
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])

            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='blues',
                    color=[],
                    size=20,
                    colorbar=dict(
                        thickness=10,
                        title='# Connections',
                        xanchor='left',
                        titleside='right'
                    ),
                    line=dict(width=0)))

            for node in G.nodes():
                x, y = G.nodes[node]['pos']
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])

            for node, adjacencies in enumerate(G.adjacency()):
                node_trace['marker']['color']+=tuple([len(adjacencies[1])]) #Coloring each node based on the number of connections 
                node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
                node_trace['text']+=tuple([node_info])
            
            #Plot the final figure
            fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Hover over node to see city information',
                            title_x=0.08,
                            titlefont=dict(size=25),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        st.plotly_chart(fig, use_container_width=True) #Show the graph in streamlit

def display_conclusion():
    st.title('Conclusion')
    st.write('Insert conclusion text here')

if __name__ == '__main__':
    main()
