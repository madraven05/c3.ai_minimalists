import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
import contextily as ctx

path = "json/State_polygon_vulnerability.json"

def plot_state_social_vul(path):
    
    # plt.savefig(path, format='PNG')
    df = gpd.read_file(path)
    df = df.to_crs(epsg=3857)

    df['coords'] = df['geometry'].apply(lambda x: x.representative_point().coords[:])
    df['coords'] = [coords[0] for coords in df['coords']]

    # df.plot()

    # Adding Colours to the states
    colors = []
    for vul in df['Social_vul']:
        # print(vul)
        if round(vul,3) <= 0.239 and round(vul,3) >= 0.105:
            colors.append("#1f418f") #
        elif round(vul,3) <= 0.359 and round(vul,3) >= 0.240:
            colors.append("#24adbd") #
        elif round(vul,3) <= 0.535 and round(vul,3) >= 0.360:
            colors.append("#16d933") #
        elif round(vul,3) <= 0.777 and round(vul,3) >= 0.536:
            colors.append("#cfd916") # 
        elif round(vul,3) <= 1.0 and round(vul,3) >= 0.778:
            colors.append("#e32f0b") # Red
        else:
            print(vul)

    # print(colors)
    df['colors'] = colors

    ax = df.plot(color=df['colors'], figsize=(10,10), edgecolor='black', linewidth=0.7)
    for idx, row in df.iterrows():
        plt.annotate(s=row['NAME'], xy=row['coords'],
                    horizontalalignment='center', size=8)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    # Legend Elements
    legend_elements = [
        Patch(facecolor='#1f418f', label='0.105-0.239'),
        Patch(facecolor='#24adbd', label='0.240-0.359'),
        Patch(facecolor='#16d933', label='0.360-0.535'),
        Patch(facecolor='#cfd916', label='0.536-0.777'),
        Patch(facecolor='#e32f0b', label='0.778-1.000')
    ]
    plt.legend(handles=legend_elements, title="Social Vulnerability")
    
    plt.show()
    

def plot_state_health_vul(path):
    pass

plot_state_social_vul(path)