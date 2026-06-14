// JavaScript Document
function createax(figureContainerId,axId,backgroundColor,edgeSize,edgeColor){
	figure = document.getElementById(figureContainerId);
	ax = document.createElement('div')
	ax.id = axId;
	ax.className = 'ax-container';
	ax.style.backgroundColor = backgroundColor;
	ax.style.border = `${edgeSize}px solid ${edgeColor}`;
	figure.appendChild(ax)
}