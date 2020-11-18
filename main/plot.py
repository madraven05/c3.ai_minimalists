import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import geopandas as gpd
import contextily as ctx
import seaborn as sns

path_state_vul = "../json/State_polygon_vulnerability_.json"
path_road_vul = "../json/State_road_vulnerability_upd.json"
path_centroid_vul = "../json/state_centroid_vulnerability.json"
path_air_transport_vul = "../json/State_air_transport_vulnerab.json"
path_airport_vul = "../json/State_airport_vulnerability_.json"
path_heatmaps = "../air_matrix_input1.csv"


def plot_state_social_vul(path):
    '''
    Plot Social Vulnerability
    '''
    
    # Read json file and set epsg to 3857
    df = gpd.read_file(path_state_vul)
    df = df.to_crs(epsg=3857)

    # Adding Colours to the states
    colors = []
    for vul in df['Social_vul']:
        # print(vul)
        if round(vul,3) <= 0.104 and round(vul,3) >= 0.0:
            colors.append("#c78ff7") # Light Purple
        elif round(vul,3) <= 0.239 and round(vul,3) >= 0.105:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.359 and round(vul,3) >= 0.240:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 0.535 and round(vul,3) >= 0.360:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 0.777 and round(vul,3) >= 0.536:
            colors.append("#cfd916") # Yellow
        elif round(vul,3) <= 1.0 and round(vul,3) >= 0.778:
            colors.append("#e32f0b") # Red
        else:
            print(vul)

    # print(colors)
    df['colors'] = colors

    ax = df.plot(color=df['colors'], figsize=(10,10), edgecolor='black', linewidth=0.7)
    
    # Adding names to the states
    df['coords'] = df['geometry'].apply(lambda x: x.representative_point().coords[:])
    df['coords'] = [coords[0] for coords in df['coords']]

    for idx, row in df.iterrows():
        plt.annotate(text=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Legend Elements
    legend_elements = [
        Patch(facecolor='#c78ff7', label='0.000-0.104'),
        Patch(facecolor='#1f418f', label='0.105-0.239'),
        Patch(facecolor='#24adbd', label='0.240-0.359'),
        Patch(facecolor='#16d933', label='0.360-0.535'),
        Patch(facecolor='#cfd916', label='0.536-0.777'),
        Patch(facecolor='#e32f0b', label='0.778-1.000')
    ]
    plt.legend(handles=legend_elements, title="Social Vulnerability")
    
    plt.show()
    

def plot_state_health_vul():
    ''''
    Plot Health Vulnerability
    '''
    
    # Read json file and set epsg to 3857
    df = gpd.read_file(path_state_vul)
    df = df.to_crs(epsg=3857)

    # Adding Colours to the states
    colors = []
    for vul in df['health_vul']:
        # print(vul)
        if round(vul,3) <= 0.104 and round(vul,3) >= 0.0:
            colors.append("#772eff") # Light Purple
        elif round(vul,3) <= 0.239 and round(vul,3) >= 0.105:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.359 and round(vul,3) >= 0.240:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 0.535 and round(vul,3) >= 0.360:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 0.777 and round(vul,3) >= 0.536:
            colors.append("#cfd916") # Yellow
        elif round(vul,3) <= 1.0 and round(vul,3) >= 0.778:
            colors.append("#e32f0b") # Red
        else:
            print(vul)

    # print(colors)
    df['colors'] = colors

    ax = df.plot(color=df['colors'], figsize=(10,10), edgecolor='black', linewidth=0.7)
    
    # Adding names to the states
    df['coords'] = df['geometry'].apply(lambda x: x.representative_point().coords[:])
    df['coords'] = [coords[0] for coords in df['coords']]

    for idx, row in df.iterrows():
        plt.annotate(text=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Legend Elements
    legend_elements = [
        Patch(facecolor='#772eff', label='0.000-0.104'),
        Patch(facecolor='#1f418f', label='0.105-0.239'),
        Patch(facecolor='#24adbd', label='0.240-0.359'),
        Patch(facecolor='#16d933', label='0.360-0.535'),
        Patch(facecolor='#cfd916', label='0.536-0.777'),
        Patch(facecolor='#e32f0b', label='0.778-1.000')
    ]
    plt.legend(handles=legend_elements, title="Health Vulnerability")
    
    plt.show()



def plot_state_total_vul():

    '''
    Plot Total Vulnerability
    '''

    # Read json file and set epsg to 3857
    df = gpd.read_file(path_state_vul)
    df = df.to_crs(epsg=3857)

    # Adding Colours to the states
    colors = []
    for vul in df['Total_vul']:
        # print(vul)
        if round(vul,3) <= 0.104 and round(vul,3) >= 0.0:
            colors.append("#772eff") # Light Purple
        elif round(vul,3) <= 0.239 and round(vul,3) >= 0.105:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.359 and round(vul,3) >= 0.240:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 0.535 and round(vul,3) >= 0.360:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 0.777 and round(vul,3) >= 0.536:
            colors.append("#cfd916") # Yellow
        elif round(vul,3) <= 1.0 and round(vul,3) >= 0.778:
            colors.append("#e32f0b") # Red
        else:
            print(vul)

    # print(colors)
    df['colors'] = colors

    ax = df.plot(color=df['colors'], figsize=(10,10), edgecolor='black', linewidth=0.7)
    
    # Adding names to the states
    df['coords'] = df['geometry'].apply(lambda x: x.representative_point().coords[:])
    df['coords'] = [coords[0] for coords in df['coords']]

    for idx, row in df.iterrows():
        plt.annotate(text=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Legend Elements
    legend_elements = [
        Patch(facecolor='#772eff', label='0.000-0.104'),
        Patch(facecolor='#1f418f', label='0.105-0.239'),
        Patch(facecolor='#24adbd', label='0.240-0.359'),
        Patch(facecolor='#16d933', label='0.360-0.535'),
        Patch(facecolor='#cfd916', label='0.536-0.777'),
        Patch(facecolor='#e32f0b', label='0.778-1.000')
    ]
    plt.legend(handles=legend_elements, title="Total Vulnerability")
    
    plt.show()


def plot_state_road_vul():
    '''
    Plot Road Networks Vulnerability
    '''
    
    # Read json file and set epsg to 3857
    road_df = gpd.read_file(path_road_vul)
    road_df = road_df.to_crs(epsg=3857)
    state_df = gpd.read_file(path_state_vul)
    state_df = state_df.to_crs(epsg=3857)
    centroid_df = gpd.read_file(path_centroid_vul)
    centroid_df = centroid_df.to_crs(epsg=3857)

    # Adding colour to the paths
    colors = []
    for vul in road_df['wt_ij']:
        # print(vul)
        if round(vul,3) <= 0.094 and round(vul,3) >= 0.0:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.194 and round(vul,3) >= 0.095:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 0.308 and round(vul,3) >= 0.195:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 0.497 and round(vul,3) >= 0.309:
            colors.append("#cfd916") # Yellow
        elif round(vul,3) <= 1 and round(vul,3) >= 0.498:
            colors.append("#e32f0b") # Red    
        else:
            print(vul)

    road_df['colors'] = colors

    # Adding colors for the centroids
    colors_centroid = []
    for vul in state_df['Total_vul']:
        if round(vul,3) <= 0.094 and round(vul,3) >= 0.0:
            colors_centroid.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.194 and round(vul,3) >= 0.095:
            colors_centroid.append("#24adbd") # Light Blue
        elif round(vul,3) <= 0.308 and round(vul,3) >= 0.195:
            colors_centroid.append("#16d933") # Light Green
        elif round(vul,3) <= 0.497 and round(vul,3) >= 0.309:
            colors_centroid.append("#cfd916") # Yellow
        elif round(vul,3) <= 1 and round(vul,3) >= 0.498:
            colors_centroid.append("#e32f0b") # Red    
        else:
            print(vul)

    # print(len(colors_centroid))
    centroid_df['colors'] = colors_centroid


    # Legend Elements
    legend_elements_network = [
        Line2D([0], [0], color='#24adbd', lw=1, label='0.000 - 0.094'),
        Line2D([0], [0], color='#1f418f', lw=1, label='0.095 - 0.194'),
        Line2D([0], [0], color='#16d933', lw=1, label='0.195 - 0.308'),
        Line2D([0], [0], color='#cfd916', lw=1, label='0.309 - 0.497'),
        Line2D([0], [0], color='#e32f0b', lw=1, label='0.498 - 1'),
    ]

    legend_elements_centroid = [
        Line2D([0], [0], marker='o',color='white', label='0.000 - 0.094',markerfacecolor='#1f418f', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='0.095 - 0.194',markerfacecolor='#24adbd', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='0.195 - 0.308',markerfacecolor='#16d933', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='0.309 - 0.497',markerfacecolor='#cfd916', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='0.498 - 1',markerfacecolor='#e32f0b', markersize=15),   
    ]


    
    base = state_df.boundary.plot(color='black', linewidth=0.8)
    
    # Adding names to the states
    state_df['coords'] = state_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    state_df['coords'] = [coords[0] for coords in state_df['coords']]

    for idx, row in state_df.iterrows():
        plt.annotate(text=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)

    # Plotting
    ctx.add_basemap(base, source=ctx.providers.Stamen.TerrainBackground)
    centroid_plot = centroid_df.plot(ax=base, color=centroid_df['colors'], linewidth=4)
    road_plot = road_df.plot(ax=centroid_plot,color=road_df['colors'], figsize=(10,10), linewidth=2)
    legend_network = plt.legend(handles=legend_elements_network, title="Road Networks Vulnerability", loc='lower right')
    legend_centroid = plt.legend(handles=legend_elements_centroid, title="Total Vulnerability", loc='upper right')
    road_plot.add_artist(legend_network)
    road_plot.add_artist(legend_centroid)
    plt.show()


def plot_air_transport_vul():
    '''
    Plot Air Transport Network Vulnerability

    NOTE - This doesn't change after updating the csv files. 
    '''
    air_transport_df = gpd.read_file(path_air_transport_vul)
    airport_df = gpd.read_file(path_airport_vul)
    state_df = gpd.read_file(path_state_vul)
    state_df = state_df.to_crs(epsg=3857)
    airport_df = airport_df.to_crs(epsg=3857)
    air_transport_df = air_transport_df.to_crs(epsg=3857)

    # Adding colour to the air networks
    colors = []
    for vul in air_transport_df['Total_vul']:
        # print(vul)
        if round(vul,3) <= 0.315 and round(vul,3) >= 0.0:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.707 and round(vul,3) >= 0.316:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 1.255 and round(vul,3) >= 0.708:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 2.149 and round(vul,3) >= 1.256:
            colors.append("#cfd916") # Yellow
        elif round(vul,3) >= 2.150:
            colors.append("#e32f0b") # Red    
        else:
            print(vul)

    air_transport_df['colors'] = colors

    # Adding colour to the airports
    colors = []
    for vul in airport_df['Total_vul']:
        # print(vul)
        if round(vul,3) <= 0.315 and round(vul,3) >= 0.0:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.707 and round(vul,3) >= 0.316:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 1.255 and round(vul,3) >= 0.708:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 2.149 and round(vul,3) >= 1.256:
            colors.append("#cfd916https://web.whatsapp.com/") # Yellow
        elif round(vul,3) >= 2.150:
            colors.append("#e32f0b") # Red    
        else:
            print(vul)

    airport_df['colors'] = colors

    base = state_df.boundary.plot(color='black', linewidth=0.8)
    
    # Adding names to the states
    state_df['coords'] = state_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    state_df['coords'] = [coords[0] for coords in state_df['coords']]

    for idx, row in state_df.iterrows():
        plt.annotate(name=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)


    # Legend Elements
    legend_elements_network = [
        Line2D([0], [0], color='#1f418f', lw=1, label='0.000 - 0.315'),
        Line2D([0], [0], color='#24adbd', lw=1, label='0.316 - 0.707'),
        Line2D([0], [0], color='#16d933', lw=1, label='0.708 - 1.255'),
        Line2D([0], [0], color='#cfd916', lw=1, label='1.256 - 2.149'),
        Line2D([0], [0], color='#e32f0b', lw=1, label='2.150 - '),
    ]

    # Plotting
    ctx.add_basemap(base, source=ctx.providers.Stamen.TerrainBackground)
    airport_plot = airport_df.plot(ax=base, color=airport_df['colors'], linewidth=4)
    air_transport_df = air_transport_df.plot(ax=airport_plot,color=air_transport_df['colors'], figsize=(10,10), linewidth=2)

    plt.legend(handles=legend_elements_network, title="Air Transport Networks Vulnerability", loc='lower right')
    plt.show()

    
    
    


def plot_airport_vul():
    '''
    Plot Airport Vulnerability
    NOTE - This doesn't change after updating the csv files.
    '''

    airport_df = gpd.read_file(path_airport_vul)
    state_df = gpd.read_file(path_state_vul)
    state_df = state_df.to_crs(epsg=3857)
    airport_df = airport_df.to_crs(epsg=3857)

    # Adding colour to the airports
    colors = []
    for vul in air_transport_df['Total_vul']:
        # print(vul)
        if round(vul,3) <= 0.315 and round(vul,3) >= 0.0:
            colors.append("#1f418f") # Dark Blue
        elif round(vul,3) <= 0.707 and round(vul,3) >= 0.316:
            colors.append("#24adbd") # Light Blue
        elif round(vul,3) <= 1.255 and round(vul,3) >= 0.708:
            colors.append("#16d933") # Light Green
        elif round(vul,3) <= 2.149 and round(vul,3) >= 1.256:
            colors.append("#cfd916") # Yellow
        elif round(vul,3) >= 2.150:
            colors.append("#e32f0b") # Red    
        else:
            print(vul)

    airport_df['colors'] = colors

    base = state_df.boundary.plot(color='black', linewidth=0.8)
    
    # Adding names to the states
    state_df['coords'] = state_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    state_df['coords'] = [coords[0] for coords in state_df['coords']]

    for idx, row in state_df.iterrows():
        plt.annotate(text=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)

    # Legned Handle
    legend_elements_centroid = [
        Line2D([0], [0], marker='o',color='white', label='0.000 - 0.315',markerfacecolor='#1f418f', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='0.316 - 0.707',markerfacecolor='#24adbd', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='0.708 - 1.255',markerfacecolor='#16d933', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='1.256 - 2.149',markerfacecolor='#cfd916', markersize=15),
        Line2D([0], [0], marker='o',color='white', label='2.150 - ',markerfacecolor='#e32f0b', markersize=15),   
    ]

    # Plotting
    ctx.add_basemap(base, source=ctx.providers.Stamen.TerrainBackground)
    airport_plot = airport_df.plot(ax=base, color=airport_df['colors'], linewidth=4)

    plt.legend(handles=legend_elements_centroid, title="Airport Vulnerability", loc='lower right')
    plt.show()


def plot_heatmaps(month):
    '''
    Plot Heatmaps for Air Networks Vulnerability
    This only makes heatmaps for the months of May, July and October
    This is just for visualisation purposes

    Parameters -> 
    month = "may", "july" or "october" (case insensetive)
    '''
    
    data = pd.read_csv(path_heatmaps)
    data.set_index('i', inplace=True)

    if month.lower() == "october":
        df = data.pivot_table(values='Transport', index="i", columns='j',fill_value=0)
    elif month.lower() == "july":
        df = data.pivot_table(values='Transport_july', index="i", columns='j',
           fill_value=0)
    elif month.lower() == "may":
        df = data.pivot_table(values='Transport_May', index="i", columns='j',fill_value=0)
    else:
        print("Invalid month!")
        return 

    # Plotting
    plt.subplots(figsize=(35,25))
    sns.heatmap(df,cmap='icefire',linewidths=.5, annot_kws={"size": 10})
    sns.set(font_scale=0.4)
    plt.suptitle(month.upper(), fontsize='large')
    plt.show()