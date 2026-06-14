function createscale(mapContainerId,map,scaletype,sacleid,ratio,posx = null,posy = null,numticks=null,
					  labellist=null,borderratio=1){
	// 添加比例尺控件
	var mapContainer = document.getElementById(mapContainerId);
	// 获取地图容器的实时宽度和高度
	var containerWidth = mapContainer.offsetWidth;
	var containerHeight = mapContainer.offsetHeight;
	
	var scaleControl = L.control.scale({
		maxWidth: containerWidth*0.04*ratio, // 设置最大宽度
		updateWhenIdle: false, // 不在地图移动或缩放时更新比例尺
		metric: true, // 只显示公制单位
		imperial: false // 不显示英制单位
	}).addTo(map);
    var customScaleContainer = document.createElement('div');
	
    customScaleContainer.id = sacleid;
	if (scaletype === "line") {
    	customScaleContainer.className = 'custom-scale-line';
	}
	else if (scaletype === "streaky") {
		customScaleContainer.className = 'custom-scale-streaky';
	}
	mapContainer.appendChild(customScaleContainer);
	map.on('moveend zoomend', function () {
		var scaleText = scaleControl._container.querySelector('.leaflet-control-scale-line').textContent || scaleControl._container.querySelector('.leaflet-control-scale-line').innerText;
		// 获取比例尺的最新数据
		var scaleWidth = scaleControl._container.querySelector('.leaflet-control-scale-line').offsetWidth;
		var scaleHeight =scaleControl._container.querySelector('.leaflet-control-scale-line').offsetHeight;
		if (posx !== null) { 
			customScaleContainer.style.left=`${posx-scaleWidth}px`;
		}
		if (posy !== null) {
			customScaleContainer.style.bottom=`${posy}px`;
		}
		customScaleContainer.style.bottom=`${posy}px`;
		var lastNumberIndex = scaleText.search(/\D/); // 查找首个非数字字符的位置
		if (lastNumberIndex !== -1) {
			var numberPart = parseFloat(scaleText.substring(0, lastNumberIndex)); // 提取并解析数字部分
			var unitPart = scaleText.substring(lastNumberIndex).trim(); // 提取单位部分
		}
		// 调用函数生成自定义比例尺
		if (scaletype === "line") {
			createlineScale(sacleid,numberPart, unitPart, scaleWidth, scaleHeight,ratio,numticks,labellist,borderratio);
		}
		else if (scaletype === "streaky") {
			createstreakyScale(sacleid, numberPart, unitPart, scaleWidth, scaleHeight,ratio,numticks,labellist,borderratio);
		}
	});
	// 重置地图方向
	function resetMapBearing() {
		map.setBearing(0);
	};
	map.fire('moveend');	
	
}
