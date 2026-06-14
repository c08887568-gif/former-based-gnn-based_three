// JavaScript Document

function createcompass(containerId, compassId, type) {
    var container = document.getElementById(containerId);
    var compassContainer = document.createElement('div');
    compassContainer.id = compassId;
    compassContainer.className = 'compass-container';
    container.appendChild(compassContainer);
    var compassicon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    compassicon.setAttribute("class", "compass-icon");
    compassicon.setAttribute("aria-hidden", "true");

    // 创建 <use> 元素
    var useElement = document.createElementNS("http://www.w3.org/2000/svg", "use");
    if (type === 1) {
        useElement.setAttributeNS("http://www.w3.org/1999/xlink", "href", "#icon-zhibeizhen");
    } else if (type === 2) {
        useElement.setAttributeNS("http://www.w3.org/1999/xlink", "href", "#icon-zhibeizhen1");
    } else if (type == 3){
        useElement.setAttributeNS("http://www.w3.org/1999/xlink", "href", "#icon-zhibeizhen2");	
    } else if (type == 4){
		useElement.setAttributeNS("http://www.w3.org/1999/xlink", "href", "#icon-zhibeizhen-white");
	} else if (type == 5){
		useElement.setAttributeNS("http://www.w3.org/1999/xlink", "href", "#icon-zhibeizhen1-white");
	} else if (type == 6){
		useElement.setAttributeNS("http://www.w3.org/1999/xlink", "href", "#icon-zhibeizhen2-white");
	}


    // 将 <use> 元素添加到 <svg> 元素中
    compassicon.appendChild(useElement);
    compassContainer.appendChild(compassicon);
}

function compassinitialize(id,left, bottom, size,backgroundcolor,edgecolor,edgesize) {
    var container = document.getElementById(id);
	if (left!==null) {
		container.style.left = `${left}px`;
	}
    if (bottom!==null) {
		container.style.bottom = `${bottom}px`;
	}
    if (size!==null) {
		container.style.width = `${size}px`;
	}
	container.style.backgroundColor = backgroundcolor;
	container.style.border = `${edgesize}px solid ${edgecolor}`;
}