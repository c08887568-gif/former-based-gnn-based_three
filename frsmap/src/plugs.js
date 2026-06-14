// JavaScript Document
// JavaScript Document
function legenddraggable(containerId) {
    $(document).ready(function() {
        var $legendContainer = $('#' + containerId);
        var originalHeight = $legendContainer.height();
        // 直接模拟拖动结束，锁定高度
        lockHeight();
        $legendContainer.draggable({
            containment: 'parent', // 可选，限制拖动范围为父元素内 
            start: function(event, ui) {
                // 锁定图片或SVG的最大高度
                var $images = $legendContainer.find('> .legend-item > img, > .legend-item > svg');
                $images.each(function() {
                    var $this = $(this);
                    var originalImageHeight = $this.height();
                    $this.css('max-height', originalImageHeight);
                });
            },
            stop: lockHeight
        }); // 启用拖动功能
        // 定义lockHeight函数，用于锁定高度
        function lockHeight(event, ui) {
            // 恢复原始高度
            var originalHeight = $legendContainer.height();
            $legendContainer.height(originalHeight);
            // 恢复图片或SVG的高度
            var $images = $legendContainer.find('> .legend-item > img, > .legend-item > svg');
            $images.css('max-height', '');
        }
    });
}
function legendresizable(containerId) {
    $(document).ready(function() {
        var $legendContainer = $('#' + containerId);
        var originalHeight = $legendContainer.height();
        $legendContainer.resizable({ // 启用缩放功能
            handles: 'se', // 只允许从右下角进行缩放
            minHeight: 80, // 设置最小高度
            minWidth: 150, // 设置最小宽度
            resize: function(event, ui) { // 监听 resize 事件
                adjustFontSize($legendContainer, ui.size.height);
            }
        });
        // 初始化时模拟触发一次resize事件，使用当前的legend-container整体高度
        adjustFontSize($legendContainer, originalHeight);
    });
}
function compassdraggable(containerId) {
    $(document).ready(function() {
        $("#" + containerId).draggable({
            containment: 'parent' // 可选，限制拖动范围为父元素内
        });
    });
}
function compassresizable(containerId) {
    $(document).ready(function() {
        $("#" + containerId).resizable({
            aspectRatio: true // 设置为true可以保持原始宽高比进行缩放
        });
    });
}