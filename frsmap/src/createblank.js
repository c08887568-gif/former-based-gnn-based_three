// JavaScript Document
function createblank(figureContainerId){
	figure = document.getElementById(figureContainerId);
	ax = document.createElement('div')
	ax.className = 'blank-container';
	figure.appendChild(ax)
}