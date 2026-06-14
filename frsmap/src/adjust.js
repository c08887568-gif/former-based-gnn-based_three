// JavaScript Document
//定义adjustFontSize函数，用于自适应
function adjustFontSize(legendContainer,currentHeight) {
	var scaleFactor = (currentHeight / 100) ** 1.01; // 这里的1.5是一个调整因子，可以根据实际效果调整
	var fontSize = Math.max(scaleFactor * 10, scaleFactor * 15); // 假设基础字体大小为16px，这里设置了最小字体大小为10px
	$('.legend-item', legendContainer).css('font-size', fontSize + 'px'); // 更新字体大小
	var padding_top = currentHeight*0.4; // 根据高度计算title空间
	var padding = currentHeight*0.04; // 根据高度计算title空间
	legendContainer.css({
		'padding': padding+'px',
		'padding-top': padding_top+'px',
		'padding-left': padding*3+'px',
		'padding-right': padding*3+'px'
	});
	var fontSize_title = fontSize*1.2; // 根据高度计算字体大小，这里设置了最小字体大小为10px
	$('.legend-title', legendContainer).css('font-size', fontSize_title + 'px'); // 更新字体大小
	var gap = Math.max(10, currentHeight *0.2); 
	$('.legend-item', legendContainer).css('gap', gap + 'px');

}