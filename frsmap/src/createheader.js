// JavaScript Document
function createheader(mapContainerId,textContent, fontSize = null, fontFamily = 'Times New Roman',
					  fontColor = '#000',backgroundColor = '#fff') {
    var mapContainer = document.getElementById(mapContainerId);
	mapContainerheight = mapContainer.offsetHeight
    var header = document.createElement('div');
    header.className = 'header-bar';
    mapContainer.appendChild(header);
    var mapContainerHeight = mapContainer.clientHeight;
    if (fontSize === null || fontSize === undefined) { 
		header.style.height = `${mapContainerHeight*0.04}px`;
        header.style.fontSize = `${mapContainerHeight * 0.04}px`;
    } else {
		header.style.height = `${fontSize}px`;
        header.style.fontSize = `${fontSize}px`;
    }

    header.style.fontFamily = "'" + fontFamily + "', Arial, sans-serif";
    header.textContent = textContent;
	header.style.color = fontColor; // 字体颜色
    header.style.backgroundColor = backgroundColor; // 背景颜色
}