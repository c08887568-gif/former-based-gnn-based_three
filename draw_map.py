import os
import glob
import pandas as pd
import frsmap.geoplotlib as plt
def plot_trajectories(excel_file_path, output_folder):
    # 读取Excel文件
    df = pd.read_excel(excel_file_path)
    # 数据类型转换及处理
    data1 = df[['Longitude', 'Latitude']]
    label1 = df['Predicted_Label']
    fig = plt.figure(num=1,figsize=(12,10),width_ratio=0.99,height_ratio=0.99,dpi=100,backgroundolor='#fff',edgecolor='transparent',edgesize=2)
    url="https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=YOUR_MAPBOX_ACCESS_TOKEN"
    ax = fig.add_subplot(nrows=1,ncols=1,index=1,backgroundcolor='#fff',edgecolor='transparent',edgesize=2)
    ax.geomap(url,center=[34.919378, 118.987609],zoomlevel=17)
    ax.geoscatter(data1,label1,["#FF0000","#39FF14"],size=3,centertype='unweighted')
    ax.compass(compass_type=6,size=150,left='null',bottom='null',backgroundcolor='transparent',edgecolor='transparent',edgesize=1)
    ax.scale(scale_type='line',ratio=4,borderratio=2,left=50,bottom=50,numticks=2,labellist=[0,2])
    ax.addgraticule(numticks=5,ratio=1.5)
    output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(excel_file_path))[0]}")
    plt.save_as_html(output_file+'.html')
    plt.save_as_img(output_file+'.png')
# 遍历目录结构
root_dir = "./outputs"
for model_dir in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_dir)
    if os.path.isdir(model_path):
        for dataset_dir in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_dir)
            if os.path.isdir(dataset_path):
                trajectories_path = os.path.join(dataset_path, "trajectories")
                if os.path.exists(trajectories_path):
                    # 获取所有.xlsx文件
                    excel_files = glob.glob(os.path.join(trajectories_path, "*.xlsx"))
                    
                    # 创建输出目录
                    output_folder = os.path.join(dataset_path, "RemoteSensingImages")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    
                    # 处理每个文件
                    for excel_file in excel_files:
                        plot_trajectories(excel_file, output_folder)