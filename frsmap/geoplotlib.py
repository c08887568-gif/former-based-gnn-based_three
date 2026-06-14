from .utils import convert_data,to_html,remove_dependencies,to_img
import numpy as np
import os
class geoplotlib:
    figures = []
    figure_now = None
    @classmethod
    def geomap(cls,url, center, zoomlevel=14, maxzoom=20):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        ax=fig.axes[-1]
        ax.geomap(url, center, zoomlevel, maxzoom)
        fig.update_html()
    @classmethod
    def geoscatter(cls, coordinates, labels, colormap, size,centertype='unweighted'):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes) == 0:
            fig.axes.append(Axes(1))
        ax = fig.axes[-1]
        center = list(np.mean(np.array(coordinates), axis=0))
        url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=YOUR_MAPBOX_ACCESS_TOKEN"
        if ax._geomap is None:
            ax.geomap(url, center[::-1], zoomlevel=14, maxzoom=20)
        ax.geoscatter(coordinates, labels, colormap, size,centertype)
        fig.update_html()
    @classmethod
    def figure(cls,num=1,figsize=(12,8),width_ratio=0.8,height_ratio=0.9,dpi=100,backgroundolor='#fff',edgecolor='#000',edgesize=2):
        fig = Figure(num,figsize,width_ratio,height_ratio,dpi,backgroundolor,edgecolor,edgesize)
        cls.figures.append(fig)
        cls.figure_now = fig
        return cls.figure_now
    @classmethod
    def delfig(cls,index):
        index = index-1
        if 0 <= index < len(cls.figures):
            if cls.figures[index] == cls.figure_now:
                cls.figure_now = None
            cls.figures.pop(index) 
    @classmethod
    def subplots(cls,nrows, ncols, num=1,figsize=(12,8),width_ratio=0.8,height_ratio=0.8,dpi=100,figure_backgroundolor='#fff',
                 figure_edgecolor='#000',figure_edgesize=2,ax_backgroundcolor='#fff',ax_edgecolor='#000',ax_edgesize=2):
        figure_now = Figure(num,figsize,width_ratio,height_ratio,dpi,figure_backgroundolor,figure_edgecolor,figure_edgesize)
        cls.figures.append(figure_now)
        cls.figure_now = figure_now
        fig = cls.figure_now
        for index in range(nrows*ncols):
            index = index+1
            fig.add_subplot(nrows,ncols,index,ax_backgroundcolor,ax_edgecolor,ax_edgesize)
        fig.update_html()
        return fig , fig.axes
    @classmethod
    def legend(cls,legenditems,title='Legend',left='null',bottom='null',width='null',height='null',backgroundcolor='#fff',
                  edgecolor='#000',edgesize=1,fontsize='null',fontfamily='Times New Roman',draggable=True,resizable=True):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        ax=fig.axes[-1]
        fig.sublegend(legenditems,title,left,bottom,width,height,backgroundcolor,
                  edgecolor,edgesize,fontsize,fontfamily,draggable,resizable)
    @classmethod
    def line2D(cls,label,iconcolor,fontcolor='#000'):
        line = dict(             
            icon=dict(
                ltype='line',
                style=dict(
                    backgroundColor=iconcolor,
                ),   
            ),
            span=dict(
                textContent=label,
                style=dict(
                    color=fontcolor,
                ),    
            ),
        )
        return line
    @classmethod
    def block2D(cls,label,iconcolor,fontcolor='#000'):
        block = dict(             
            icon=dict(
                ltype='block',
                style=dict(
                    backgroundColor=iconcolor,
                ), 
            ),
            span=dict(
                textContent=label,
                style=dict(
                    color=fontcolor,
                ),
            ),
        )
        return block
    @classmethod
    def point2D(cls,label,iconcolor,fontcolor='#000'):
        point = dict(             
            icon=dict(
                ltype='point',
                style=dict(
                    backgroundColor=iconcolor,
                ),  
            ),
            span=dict(
                textContent=label,
                style=dict(
                    color=fontcolor,
                ), 
            ),
        )
        return point
    @classmethod
    def compass(cls,compass_type=1,left='null',bottom='null',size='null',backgroundcolor='transparent',edgecolor='#000',edgesize=1):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        ax=fig.axes[-1]
        fig.subcompass(compass_type,left,bottom,size,backgroundcolor,edgecolor,edgesize)
    @classmethod
    def scale(cls,scale_type='line',ratio=1):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        ax=fig.axes[-1]
        ax.scale(scale_type,ratio)
    @classmethod
    def addgraticule(cls,numticks=5):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        ax=fig.axes[-1]
        ax.addgraticule(numticks)
    @classmethod
    def title(cls,text,fontsize='null',fontfamily='Times New Roman'):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        ax=fig.axes[-1]
        fig.subtitle(text,fontsize,fontfamily)
    @classmethod
    def show(cls,wait_time=5):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        if len(fig.axes)==0:
            fig.axes.append(Axes(1))
        fig.show(wait_time)
    @classmethod
    def save_as_img(cls,imagename,wait_time=5):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        fig.save_as_img(imagename,wait_time)
    @classmethod
    def save_as_html(cls,htmlname):
        if cls.figure_now is None:
            figure_now = Figure()
            cls.figures.append(figure_now)
            cls.figure_now = figure_now
        fig = cls.figure_now
        fig.save_as_html(htmlname)
        
class Figure:
    def __init__(self,num=1,figsize=(12,8),width_ratio=0.8,height_ratio=0.9,dpi=100,backgroundolor='#fff',edgecolor='#000',edgesize=2):
        self.num = num
        self.figsize = figsize
        self.dpi = dpi
        self.pxsize = tuple(x * dpi for x in figsize)
        self.backgroundolor = backgroundolor
        self.edgecolor = edgecolor
        self.edgesize = edgesize
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        self.body = dict(
            lid = 'main',
            style=dict(
                width=self.pxsize[0],
                height=self.pxsize[1],
                padding=int(self.pxsize[1]*0.01),
                backgroundColor=backgroundolor,
                edgeColor=edgecolor,
                edgeSize=edgesize,
                border=str(edgesize)+ 'px' + ' solid ' + edgecolor,
            ), 
            legend = None,
            compass = None,
            title = None,
        )
        self.figure = dict(
            style=dict(
                max_width=self.pxsize[0],
                max_height=self.pxsize[1],
                left=int(self.pxsize[1]*0.01),
                bottom = int(self.pxsize[1]*0.01),
                grid_column_gap=int(self.pxsize[1]*0.01),
                grid_row_gap=int(self.pxsize[1]*0.01),
                padding=0,
            ),
            map_Container=dict(
                style=dict(
                    width=int(self.pxsize[0]*width_ratio),
                    height=int(self.pxsize[1]*height_ratio),
                    border='0px solid #000',
                    backgroundColor='#fff',
                ),
            ),
            geomap=dict(
                style=dict(
                    width=int(self.pxsize[0]*width_ratio-0.05*self.pxsize[1]*height_ratio),
                    height=int(self.pxsize[1]*height_ratio-0.05*self.pxsize[1]*height_ratio),
                    border='0px solid #000',
                ),
            ),
        )
        self.axes = []
        self.html = None
        self.img = None
        self.is_change = False
    def add_subplot(self,nrows,ncols,index,backgroundcolor='#fff',edgecolor='#000',edgesize=2):
        if len(self.axes) < nrows*ncols :
            self.figure['map_Container']['style']['width'] = int(self.pxsize[0]*self.width_ratio/nrows)
            self.figure['map_Container']['style']['height'] = int(self.pxsize[1]*self.height_ratio/ncols)
            self.figure['geomap']['style']['width'] = int(self.pxsize[0]*self.width_ratio/nrows-0.05*self.pxsize[0]*self.height_ratio/ncols)
            self.figure['geomap']['style']['height'] = int(self.pxsize[1]*self.height_ratio/ncols-0.05*self.pxsize[0]*self.height_ratio/ncols)
            for i in range(nrows*ncols-len(self.axes)):
                self.axes.append(None)
        assert index<=len(self.axes)
        self.axes[index-1] = Axes(index,backgroundcolor,edgecolor,edgesize)
        self.update_html()
        return self.axes[index-1]
    def subplots_adjust(self,left=2,bottom=2,wspace=10,hspace=10,padding=0,edgecolor='#000',edgesize=2):
        self.figure['style']['left'] = left
        self.figure['style']['bottom'] = bottom
        self.figure['style']['grid_column_gap'] = wspace
        self.figure['style']['grid_row_gap'] = hspace
        self.figure['style']['padding'] = padding
        self.figure['map_Container']['style']['border'] = str(edgesize)+'px'+' solid '+edgecolor
        self.update_html()
    def delaxes(self,index):
        assert index<=len(self.axes)
        self.axes[index-1] = None
        self.update_html()
    def sublegend(self,legenditems,title='Legend',left='null',bottom='null',width='null',height='null',backgroundcolor='#fff',
                  edgecolor='#000',edgesize=1,fontsize='null',fontfamily='Times New Roman',draggable=True,resizable=True):
        sublegend=dict(
            lid='legend-top',
            style=dict(
                left=left,
                bottom=bottom,
                width=width,
                height=height,
                backgroundColor=backgroundcolor,
                edgeSize=edgesize,
                edgeColor=edgecolor,
                fontSize=fontsize,
                fontFamily=fontfamily,
            ),
            title=dict(
                textContent=title,
            ),
            legenditems=legenditems,
            draggable=draggable,
            resizable=resizable,
        )
        self.body['legend'] = sublegend
        self.update_html()
    def subcompass(self,compass_type=1,left='null',bottom='null',size='null',backgroundcolor='transparent',edgecolor='#000',edgesize=1):
        subcompass=dict(
            lid='compass-top',
            style=dict(
                size=size,
                left=left,
                bottom=bottom,
                backgroundColor=backgroundcolor,
                edgeColor=edgecolor,
                edgeSize=edgesize,
            ),
            icon=dict(
                ltype=compass_type,
            ),
            draggable=True,
            resizable=True,
        )
        self.body['compass'] = subcompass
        self.update_html()
    def subtitle(self,text,fontsize='null',fontfamily='Times New Roman',fontcolor='#000',backgroundcolor='#fff'):
        title=dict(
            style=dict(
                fontSize=fontsize,
                fontFamily=fontfamily,
                fontColor=fontcolor,
                backgroundColor=backgroundcolor,
            ),
            textContent=text,
        )
        self.body['title'] = title
        self.update_html()
    def update_html(self):
        axesdict = [ax.html if ax else None for ax in self.axes]
        self.html = dict(
            body = self.body,
            figure = self.figure,
            axes = axesdict,
        )
        if len(self.axes) >0 :
            for ax in self.axes:
                if ax is not None:
                    if ax.is_change == True:
                        self.is_change = True
                        ax.is_change = False
    def show(self,wait_time=5):
        self.update_html()
        if self.img is None or self.is_change:
            final_html = to_html(self.html)
            outputfilename = './out_dasez.html'
            # 暂时保存修改后的 HTML
            with open(outputfilename, "w", encoding="utf-8") as file:
                file.write(final_html)
            img=to_img(outputfilename,wait_time)
            os.remove(outputfilename)
            self.img = img
        else:
            img = self.img
        try:
            # 如果在 IPython 中
            from IPython.display import display
            display(img)
        except ImportError:
            # 如果在 Python 脚本中
            img.show()
        self.is_change = False
    def save_as_img(self,imagename,wait_time=5):
        self.update_html()
        if self.img is None or self.is_change:
            
            final_html = to_html(self.html)
            outputfilename = './out_dasez.html'
            # 暂时保存修改后的 HTML
            with open(outputfilename, "w", encoding="utf-8") as file:
                file.write(final_html)
            img=to_img(outputfilename,wait_time)
            os.remove(outputfilename)
            self.img = img
        else:
            img = self.img
        img.save(imagename,quality=95)
        self.is_change = False
    def save_as_html(self,htmlname):
        self.update_html()
        final_html = to_html(self.html)
        # 暂时保存修改后的 HTML
        with open(htmlname, "w", encoding="utf-8") as file:
            file.write(final_html)
        self.is_change = False
            
            
class Axes:
    def __init__(self,index,backgroundcolor='#fff',edgecolor='#000',edgesize=2):
        self._id = 'ax' + str(index)
        self._index = index
        self._backgroundcolor = backgroundcolor
        self._edgecolor = edgecolor
        self._edgesize = edgesize
        self._geomap = None
        self._scale = None
        self._compass = None
        self._title = None
        self._caption = None
        self._graticule = None
        self._scatter = None
        self.html = None
        self.is_change = False
    def geomap(self,url, center, zoomlevel=14, maxzoom=20):
        self._geomap = dict(
            lid='map' + str(self._index),
            url=url,
            center=str(center),
            zoomlevel=zoomlevel,
            maxzoom=maxzoom,
        )
        self.update_html()
    def geoscatter(self, coordinates, labels,colormap, size,centertype='unweighted'):
        json_data = convert_data(coordinates,labels)
        if centertype == 'unweighted':
            coordinates=np.array(coordinates)
            # 计算最大和最小经度
            max_lon = np.max(coordinates[:, 0])
            min_lon = np.min(coordinates[:, 0])

            # 计算最大和最小纬度
            max_lat = np.max(coordinates[:, 1])
            min_lat = np.min(coordinates[:, 1])

            # 计算中心点
            center_lon = (max_lon + min_lon) / 2
            center_lat = (max_lat + min_lat) / 2

            center = [center_lon, center_lat]
        elif centertype == 'weighted': 
            center = list(np.mean(np.array(coordinates),axis=0))
        self._geomap['center'] = str(center[::-1])
        self._scatter = dict(
            data=json_data,
            colormap=colormap,
            size=size,
        )
        self.update_html()
    def scale(self,scale_type='line',ratio=1,borderratio=1,left='null',bottom='null',numticks=8,labellist=[0,2,4,8]):
        self._scale = dict(
            ltype = scale_type,
            ratio = ratio,
            left = left,
            bottom = bottom,
            numticks = numticks,
            labellist = list(labellist),
            borderratio = borderratio,
        )
        self.update_html()
    def compass(self,compass_type=1,size='null',left='null',bottom='null',backgroundcolor='transparent',edgecolor='#000',edgesize=1):
        self._compass = dict(
            lid='compass' + str(self._index),
            style=dict(
                size=size,
                left=left,
                bottom=bottom,
                backgroundColor=backgroundcolor,
                edgeColor=edgecolor,
                edgeSize=edgesize,
            ),
            icon=dict(
                ltype=compass_type,
            ),
            draggable=False,
            resizable=False,
        )
        self.update_html()
    def addgraticule(self,numticks=5,ratio=1):
        self._graticule=dict(
            numticks=numticks,
            ratio=ratio,
        )
        self.update_html()
    def set_title(self,text,fontsize='null',fontfamily='Times New Roman'):
        self._title=dict(
            style=dict(
                fontSize=fontsize,
                fontFamily=fontfamily,
            ),
            textContent=text,
        )
        self.update_html()
    def set_caption(self,text,fontsize='null',fontfamily='Times New Roman',fontcolor='#000',backgroundcolor='#fff'):
        self._caption=dict(
            textContent=text,
            style=dict(
                fontSize=fontsize,
                fontFamily=fontfamily,
                fontColor=fontcolor,
                backgroundColor=backgroundcolor,
            ), 
        )
        self.update_html()
    def update_html(self):
        self.html = dict(
            lid=self._id,
            style=dict(
                backgroundColor=self._backgroundcolor,
                edgeColor=self._edgecolor,
                edgeSize=self._edgesize,
            ),
            title=self._title,
            geomap=self._geomap,
            scale=self._scale,
            compass=self._compass,
            graticule=self._graticule,
            header=self._caption,
            scatter=self._scatter,
        )
        self.is_change = True