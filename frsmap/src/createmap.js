// JavaScript Document
// 创建 Leaflet 地图对象并设置中心点和缩放级别
function createmap(axContainerId,mapContainerId,id,center,zoomlevel,url,maxzoom) {
	var axContainer = document.getElementById(axContainerId); // 获取ax容器元素
	var mapContainer = document.createElement('div'); // 获取mapcontainer容器元素
	mapContainer.id = mapContainerId;
    mapContainer.className = 'map-container';
	axContainer.appendChild(mapContainer);
    var mapDiv = document.createElement('div'); // 获取mapDiv容器元素
    mapDiv.id = id;
    mapDiv.className = 'map';
	mapContainer.appendChild(mapDiv);
	var map = L.map(id, {
		attributionControl: false,
		zoomControl: false, // 关闭默认的缩放控件
	}).setView(center,zoomlevel);       
	// 添加地图瓦片图层
	L.tileLayer(url, {
		maxZoom: maxzoom,
	}).addTo(map);
	return map
}
	