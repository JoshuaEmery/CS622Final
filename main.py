import streamlit as st
import json
import pandas as pd
import plotly
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import scipy as sp

st.set_page_config(layout="wide")
# table from a dictionary
table = pd.DataFrame({'Temp Renton': [32,35,41,51,64,24,37],
                      'Temp Seattle': [31 ,46, 24, 21,71,46,51]}, index=['21st', '22nd', '23rd','24th', '25th','26th','27th'])
st.title("This is a streamlit web page")
st.subheader("This is a subheader")
st.header("This is a header")
st.text("This is text will create a p tag")

# streamlit can take markdown
st.markdown('**Bold** and *italic* coming from ***markdown***')
# markdown horizontal line
st.markdown('---')
st.markdown('[Link to RTC](https://www.rtc.edu)')

# latex for math uses katex https://katex.org/docs/supported.html
# have to use raw python string
st.latex(r'\begin{pmatrix}a&b\\c & d\end{pmatrix}')
# caption function
st.caption('Oooo math!')

# json
st.json(json.load(open('students.json')))

# embedding code
code = """
Console.WriteLine("Please enter a number");
int x = int.Parse(Console.ReadLine());
if(x < 18)
{
    Console.WriteLine("You cannot vote");
}
"""
code2 = """
        while self.routes:
            minNode = None
            for node in self.routes:
                if minNode is None:
                    minNode = node
                elif shortest_distance[node] < shortest_distance[minNode]:
                    minNode = node

            for childNode, distance in self.routes[minNode].items():
                if distance + shortest_distance[minNode] < shortest_distance[childNode]:
                    shortest_distance[childNode] = distance + shortest_distance[minNode]
                    predecessor[childNode] = minNode
            self.routes.pop(minNode)
"""
st.code(code,language="c#")
st.code(code2,language='python')
# write function
st.write('write is a catch all function that can output many types of data :sunglasses:')
st.write('emojis :sunglasses: :alien:')

# metric function
st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

st.table(table)
st.dataframe(table)

a=np.linspace(start=0,stop=36,num=36)
np.random.seed(25)
b=np.random.uniform(low=0.0,high=1.1,size=36)
trace=go.Scatter(x=a,y=b)
data=trace
# py.iplot(data,filename='basic')
fig = go.Figure(data)
st.plotly_chart(fig, use_container_width=True)


#Add a logo (optional) in the sidebar
logo = Image.open(r'logo.png')
st.sidebar.image(logo,  width=300)
# color 24, 20, 16

#Add the expander to provide some information about the app
with st.sidebar.expander("About the App"):
     st.write("""
        This network graph app was built by My Data Talk using Streamlit and Plotly. 
        You can use the app to quickly generate an interactive network graph with different 
        layout choices.
     """)

#Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader('',type=['csv']) #Only accepts csv file format

#Add an app title. Use css to style the title
st.markdown(""" <style> .font {                                          
    font-size:30px ; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Weighted undirected graph from .csv file</p>', 
            unsafe_allow_html=True)


#Create the network graph using networkx
if uploaded_file is not None:     
    df=pd.read_csv(uploaded_file)  
    A = list(df["StartCity"].unique())
    B = list(df["EndCity"].unique())
    node_list = set(A+B)
    G = nx.Graph() #Use the Graph API to create an empty network graph object
    
    #Add nodes and edges to the graph object
    for i in node_list:
        G.add_node(i)
    for i,j in df.iterrows():
        G.add_edges_from([(j["StartCity"],j["EndCity"])])    
 
    #Create three input widgets that allow users to specify their preferred layout and color schemes
    col1, col2, col3 = st.columns( [1, 1, 1])
    with col1:
        layout= st.selectbox('Choose a network layout',('Random Layout','Spring Layout','Shell Layout','Kamada Kawai Layout','Spectral Layout'))
    with col2:
        color=st.selectbox('Choose color of the nodes', ('Blue','Red','Green','Orange','Red-Blue','Yellow-Green-Blue'))      
    with col3:
        title=st.text_input('Add a chart title')

    #Get the position of each node depending on the user' choice of layout
    if layout=='Random Layout':
        pos = nx.random_layout(G) 
    elif layout=='Spring Layout':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif  layout=='Shell Layout':
        pos = nx.shell_layout(G)            
    elif  layout=='Kamada Kawai Layout':
        pos = nx.kamada_kawai_layout(G) 
    elif  layout=='Spectral Layout':
        pos = nx.spectral_layout(G) 

    #Use different color schemes for the node colors depending on he user input
    if color=='Blue':
        colorscale='blues'    
    elif color=='Red':
        colorscale='reds'
    elif color=='Green':
        colorscale='greens'
    elif color=='Orange':
        colorscale='orange'
    elif color=='Red-Blue':
        colorscale='rdbu'
    elif color=='Yellow-Green-Blue':
        colorscale='YlGnBu'

    #Add positions of nodes to the graph
    for n, p in pos.items():
        G.nodes[n]['pos'] = p


    #Use plotly to visualize the network graph created using NetworkX
    #Adding edges to plotly scatter plot and specify mode='lines'
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
    
    #Adding nodes to plotly scatter plot
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale=colorscale, #The color scheme of the nodes will be dependent on the user's input
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
                    title=title, #title takes input from the user
                    title_x=0.45,
                    titlefont=dict(size=25),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    st.plotly_chart(fig, use_container_width=True) #Show the graph in streamlit
