from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def create_edges_list(df):
    """
    Creates a list of edges from production companies
    
    Args:
        df: DataFrame containing 'production_companies' column
        
    Returns:
        list: List of company pairs (edges)
    """
    edges = []
    for companies in df['production_companies']:
        if len(companies) > 1:
            edges.extend(list(combinations(companies, 2)))
    return edges

def create_network_graph(df_filtered):
    """
    Creates a network graph from filtered data
    
    Args:
        df_filtered: Filtered DataFrame for specific year and production type
        
    Returns:
        networkx.Graph: The created network graph
        dict: Network statistics
    """
    # Create edges and graph
    edges = create_edges_list(df_filtered)
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Calculate statistics
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_collaborations': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }
    
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        stats['top_companies'] = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    else:
        stats['top_companies'] = []
    
    return G, stats

def create_network_plot(G, year, prod_type, stats):
    """
    Creates a network visualization plot
    
    Args:
        G: networkx Graph object
        year: Year of analysis
        prod_type: Production type
        stats: Dictionary of network statistics
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw network
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        node_size=1000,
        font_size=8,
        font_weight='bold',
        edge_color='gray',
        width=0.5,
        alpha=0.7
    )
    
    # Add title
    title = f"Production Company Network - {year} ({prod_type} Productions)\n"
    if stats['num_nodes'] > 0:
        title += (f"Companies: {stats['num_nodes']}, "
                 f"Collaborations: {stats['num_edges']}, "
                 f"Avg. Collaborations per Company: {stats['avg_collaborations']:.2f}")
    else:
        title += "No data"
    
    plt.title(title, pad=20)
    plt.tight_layout()
    
    return fig