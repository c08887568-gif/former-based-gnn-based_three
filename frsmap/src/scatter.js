// JavaScript Document
// 再次读取GeoJSON数据，假设数据中有新的大小信息
function geoscatter(map, data, colormap, pointsize) {
    var geojsonLayer = L.geoJSON(data, {
        pointToLayer: function(feature, latlng) {
            var color = colormap[feature.properties.label];

            return L.circleMarker(latlng, {
                radius: pointsize,
                fillColor: color,
                color: color,
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            });
        }
    }).addTo(map);
}
