// JavaScript Document

function createlegend(containerId, id, title = 'Legend', fontFamily = 'Times New Roman') {
    var container = document.getElementById(containerId);
    var legendContainer = document.createElement('div');
    legendContainer.id = id;
    legendContainer.className = 'legend-container';
    container.appendChild(legendContainer);

    var legendTitle = document.createElement('div');
    legendTitle.className = 'legend-title';
    legendTitle.textContent = title;
    legendTitle.style.fontFamily = "'" + fontFamily + "', Arial, sans-serif";
    legendContainer.appendChild(legendTitle);
}
function createlegenditem(legendContainerId, type, text, itemcolor, spancolor, fontFamily = 'Times New Roman') {
    var legendContainer = document.getElementById(legendContainerId);
    var legenditemContainer = document.createElement('div');
    legenditemContainer.className = 'legend-item';
    legendContainer.appendChild(legenditemContainer);

    var legenditem = document.createElement('div');
    if (type === 'line') {
        legenditem.className = 'legend-line-block';
    } else if (type === 'block') {
        legenditem.className = 'legend-color-block';
    } else if (type === 'point') {
        legenditem.className = 'legend-circle-block';
    }
    legenditem.style.backgroundColor = itemcolor;

    var legenditemspan = document.createElement('span');
    legenditemspan.textContent = text;
    legenditemspan.style.color = spancolor;
    legenditemspan.style.fontFamily = "'" + fontFamily + "', Arial, sans-serif";

    legenditemContainer.appendChild(legenditem);
    legenditemContainer.appendChild(legenditemspan);
}
function legendinitialize(id, left, bottom, width, height,backgroundcolor,edgecolor,edgesize) {
	var legendContainer = document.getElementById(id);
	if (left!==null) {
		legendContainer.style.left = `${left}px`;
	}
    if (bottom!==null) {
		legendContainer.style.bottom = `${bottom}px`;
	}
	legendContainer.style.backgroundColor = backgroundcolor;
	legendContainer.style.border = `${edgesize}px solid ${edgecolor}`;
	if (width!==null){
		if (width > 150) {
			legendContainer.style.width = `${width}px`;
		}
		else {
			legendContainer.style.width = `150px`;
		}
	}
	if (height!==null) {
		if (height > 80) {
			legendContainer.style.height = `${height}px`;
		}
		else {
			legendContainer.style.height = `80px`;
		}
	}
	else {
		legendContainer.style.height = '10%';
	}
	
	var $legendContainer = $('#' + id);
    var originalHeight = $legendContainer.height();
	adjustFontSize($legendContainer, originalHeight);
}