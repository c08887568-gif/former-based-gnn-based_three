// JavaScript Document
function createlineScale(customScaleContainerId, numberPart, unitPart, scaleWidth, 
						  scaleHeight,ratio,numticks=8,labellist=null,borderratio=1) {
	if (numticks % 2 === 0){
		hightick = Math.floor(numticks/2);
	}else{
		hightick = numticks
	}
	if (labellist === null){
		labellist =  Array.from({ length: numticks+1 }, (v, k) => k );
	}
    customScaleContainer = document.getElementById(customScaleContainerId)
	customScaleContainer.innerHTML = '';
	var fontsize= `${scaleHeight*0.8*ratio}px`;
	var tick = document.createElement('div');
	tick.classList.add('custom-scale-tick-line');
	tick.style.width = `${scaleWidth}px`;
	tick.style.height = `100%`;
	tick.style.top = `0px`;
	tick.style.borderRight= '0px solid #000';
	tick.style.borderBottom= '0px solid #000';
	customScaleContainer.appendChild(tick);
	var label = document.createElement('div');
	label.classList.add('custom-scale-label-line');
	label.style.fontSize=fontsize;
	if (labellist.includes(0)){
		label.textContent = `0${unitPart}`;
	}else{
		label.textContent = '';
	}
	tick.appendChild(label);
	customScaleContainer.appendChild(tick);
	var bordersize = scaleHeight*0.2*borderratio;
	for (var i = 0; i <=numticks-1; i++) {
		if (i === 0) {
			var tick = document.createElement('div');
			tick.classList.add('custom-scale-tick-line');
			tick.style.width = `${scaleWidth}px`;
			tick.style.height = `100%`;
			tick.style.top = `0px`;
			tick.style.borderRight= '0px solid #fff';
			tick.style.borderLeft= `${bordersize}px solid #fff`;
			tick.style.borderBottom= `${bordersize}px solid #fff`;
		}
		else if (i === 1) {
			var tick = document.createElement('div');
			tick.classList.add('custom-scale-tick-line');
			tick.style.width = `${scaleWidth}px`;
			tick.style.borderRight= `${bordersize}px solid #fff`;
			tick.style.borderLeft= `${bordersize}px solid #fff`;
			tick.style.borderBottom= `${bordersize}px solid #fff`;
			
		}
		else if((i +1) === hightick || (i +1) === numticks) {
			var tick = document.createElement('div');
			tick.classList.add('custom-scale-tick-line');
			tick.style.width = `${scaleWidth}px`;
			tick.style.height = `100%`;
			tick.style.top = `0px`;
			tick.style.borderRight= `${bordersize}px solid #fff`;
			tick.style.borderBottom= `${bordersize}px solid #fff`;
			

		}
		else{
			var tick = document.createElement('div');
			tick.classList.add('custom-scale-tick-line');
			tick.style.width = `${scaleWidth}px`;
			tick.style.borderRight= `${bordersize}px solid #fff`;
			tick.style.borderBottom= `${bordersize}px solid #fff`;
			
		}
		if (labellist.includes(i+1)){
			var label = document.createElement('div');
			label.classList.add('custom-scale-label-line');
			label.style.fontSize=fontsize;
			label.textContent = `${numberPart*(i+1)}${unitPart}`;
			tick.appendChild(label);					   
		}
		else{
			var label = document.createElement('div');
			label.classList.add('custom-scale-label-line');
			label.style.fontSize=fontsize;
			label.textContent = ``;
			tick.appendChild(label);
		}
		customScaleContainer.appendChild(tick);
	}
}
function createstreakyScale(customScaleContainerId, numberPart, unitPart, scaleWidth, 
							 scaleHeight,ratio,numticks=8,labellist=null,borderratio=1){
	if (labellist === null){
		labellist =  Array.from({ length: numticks+1 }, (v, k) => k );
	}
	customScaleContainer = document.getElementById(customScaleContainerId)
	customScaleContainer.innerHTML = '';
	var fontsize= `${scaleHeight*0.8*ratio}px`;
	var tick = document.createElement('div');
	tick.classList.add('custom-scale-tick-streaky');
	tick.style.width = `${scaleWidth}px`;
	tick.style.border = '0px'
	customScaleContainer.appendChild(tick);
	var label = document.createElement('div');
	label.classList.add('custom-scale-label-streaky');
	label.style.fontSize=fontsize;
	label.textContent = `0${unitPart}`;
	tick.appendChild(label);
	customScaleContainer.appendChild(tick);
	var bordersize = scaleHeight*0.15*borderratio;
	for (var i = 0; i <=numticks-1; i++) {
		ratio = Math.floor(i/4)+1
		if (i%2 === 0) {
			var tick = document.createElement('div');
			tick.classList.add('custom-scale-tick-streaky');
			tick.style.width = `${scaleWidth*ratio}px`;
			tick.style.backgroundColor = 'black';
			tick.style.border= `${bordersize}px solid #000`;
		}
		else{
			var tick = document.createElement('div');
			tick.classList.add('custom-scale-tick-streaky');
			tick.style.width = `${scaleWidth*ratio}px`;
			tick.style.backgroundColor = 'white';
			tick.style.border= `${bordersize}px solid #000`;
		}
		if (labellist.includes(i+1)){
			var label = document.createElement('div');
			label.classList.add('custom-scale-label-streaky');
			label.style.fontSize=fontsize;
			label.textContent = `${numberPart*(i+1)}${unitPart}`;
			tick.appendChild(label);					   
		}
		else{
			var label = document.createElement('div');
			label.classList.add('custom-scale-label-streaky');
			label.style.fontSize=fontsize;
			label.textContent = ``;
			tick.appendChild(label);
		}
		customScaleContainer.appendChild(tick);
	}
}