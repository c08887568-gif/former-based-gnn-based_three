// JavaScript Document
function toDMS(deg, isLatitude) {
    var absolute = Math.abs(deg);
    var degrees = Math.floor(absolute);
    var minutesNotTruncated = (absolute - degrees) * 60;
    var minutes = Math.floor(minutesNotTruncated);
    var seconds = Math.round((minutesNotTruncated - minutes) * 60);

    // 补零到两位
    var minutesStr = minutes.toString().padStart(2, '0');
    var secondsStr = seconds.toString().padStart(2, '0');

    var direction;
    if (isLatitude) {
        direction = deg >= 0 ? 'N' : 'S';
    } else {
        direction = deg >= 0 ? 'E' : 'W';
    }

    return degrees + "° " + minutesStr + "' " + secondsStr + "\" " + direction;
}

function addgraticule(mapContainerId,mapId,map,bottomGraticuleContainerId,topGraticuleContainerId,
					   leftGraticuleContainerId,rightGraticuleContainerId,
					   numticks,posx,posy,ratio){
	var bounds = map.getBounds();
    var mapContainer = document.getElementById(mapContainerId);
	var submap = document.getElementById(mapId);
	var mapwidth = submap.offsetWidth;
	var mapheight = submap.offsetHeight;
    var bottomGraticuleContainer = document.createElement('div');
    bottomGraticuleContainer.id = bottomGraticuleContainerId;
    bottomGraticuleContainer.className = 'custom-graticule';
    mapContainer.appendChild(bottomGraticuleContainer);
    var topGraticuleContainer = document.createElement('div');
    topGraticuleContainer.id = topGraticuleContainerId;
    topGraticuleContainer.className = 'custom-graticule';
    mapContainer.appendChild(topGraticuleContainer);
    var leftGraticuleContainer = document.createElement('div');
    leftGraticuleContainer.id = leftGraticuleContainerId;
    leftGraticuleContainer.className = 'custom-graticule';
    mapContainer.appendChild(leftGraticuleContainer);
    var rightGraticuleContainer = document.createElement('div');
    rightGraticuleContainer.id = rightGraticuleContainerId;
    rightGraticuleContainer.className = 'custom-graticule';
    mapContainer.appendChild(rightGraticuleContainer);
	bottomGraticuleContainer.innerHTML = '';
	topGraticuleContainer.innerHTML = '';
	leftGraticuleContainer.innerHTML = '';
	rightGraticuleContainer.innerHTML = '';
	bottomGraticuleContainer.style.bottom=`${posy}px`;
	//bottomGraticuleContainer.style.transform = `translate(0,65%)`; 
	topGraticuleContainer.style.top='0px';
	leftGraticuleContainer.style.top='0px';
	leftGraticuleContainer.style.transformOrigin = '0 0';
	leftGraticuleContainer.style.transform = 'rotate(90deg) translate(0, -100%)'; 
	rightGraticuleContainer.style.bottom=`${posy}px`;
	rightGraticuleContainer.style.transformOrigin = '0 100%';
	rightGraticuleContainer.style.transform = `rotate(-90deg) translate(0, ${posx}px)`; 
	var fontsize= `${mapheight*0.02*ratio}px`;
	var southwest = bounds.getSouthWest();
	var northeast = bounds.getNorthEast();
	var latInterval = (northeast.lat-southwest.lat)/numticks;
	var lngInterval = (northeast.lng-southwest.lng)/numticks;
	var verticaltickinterval=mapheight/numticks;
	var horizontaltickinterval=mapwidth/numticks;
	
	for (var i = 0; i <numticks-1; i++) {
		var tickbottom = document.createElement('div');
		tickbottom.classList.add('custom-graticule-tick');
		tickbottom.style.width = `${horizontaltickinterval}px`;
		var labelbottom = document.createElement('div');
		labelbottom.classList.add('custom-graticule-label');
		labelbottom.style.fontSize=fontsize;
		labelbottom.textContent = toDMS((i+1)*lngInterval+southwest.lng, true);
		tickbottom.appendChild(labelbottom);
		bottomGraticuleContainer.appendChild(tickbottom);
		
		var ticktop = document.createElement('div');
		ticktop.classList.add('custom-graticule-tick');
		ticktop.style.width = `${horizontaltickinterval}px`;
		var labeltop = document.createElement('div');
		labeltop.classList.add('custom-graticule-label');
		labeltop.style.fontSize=fontsize;
		labeltop.textContent = '';
		ticktop.appendChild(labeltop);
		topGraticuleContainer.appendChild(ticktop);
		
		var tickleft = document.createElement('div');
		tickleft.classList.add('custom-graticule-tick');
		tickleft.style.width = `${verticaltickinterval}px`;
		var labelleft = document.createElement('div');
		labelleft.classList.add('custom-graticule-label');
		labelleft.style.fontSize=fontsize;
		labelleft.textContent = '';
		tickleft.appendChild(labelleft);
		leftGraticuleContainer.appendChild(tickleft);
		
		var tickright = document.createElement('div');
		tickright.classList.add('custom-graticule-tick');
		tickright.style.width = `${verticaltickinterval}px`;
		var labelright = document.createElement('div');
		labelright.classList.add('custom-graticule-label');
		labelright.style.fontSize=fontsize;
		labelright.textContent =toDMS((i+1)*latInterval+southwest.lat, true);
		tickright.appendChild(labelright);
		rightGraticuleContainer.appendChild(tickright);
		
	}
}