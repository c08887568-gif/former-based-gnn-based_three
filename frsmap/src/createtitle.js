function createtitle(parentId,textContent,fontSize = null,fontFamily = 'Times New Roman',
					  fontColor = '#000',backgroundColor = '#fff') {
    var parent = document.getElementById(parentId);
	var parentstyle = window.getComputedStyle(parent);
    var firstChild = parent.firstChild;
    var title = document.createElement('div');
	title.className = 'header-bar';
    title.textContent = textContent;
	var containerHeight = parent.clientHeight;
	title.style.padding = `${containerHeight*0.01}px`;
	if (fontSize === null || fontSize === undefined) {
		title.style.height = `${containerHeight*0.06}px`;
        title.style.fontSize = `${containerHeight*0.06}px`;
    } else {
		title.style.height = `${fontSize}px`
        title.style.fontSize = `${fontSize}px`;
    }
    title.style.fontFamily = "'" + fontFamily + "', Arial, sans-serif";
    title.textContent = textContent;
	
	title.style.color = fontColor; // 字体颜色
    title.style.backgroundColor = backgroundColor; // 背景颜色
    parent.insertBefore(title, firstChild);
}