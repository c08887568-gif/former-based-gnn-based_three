# 导入Jinja2模板引擎
from jinja2 import Environment, FileSystemLoader
from bs4 import BeautifulSoup
import os
import json
import numpy as np
from PIL import Image
from io import BytesIO
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import time
# 获取当前脚本的所在目录
script_dir = os.path.dirname(__file__)

def convert_data(data, label):
    data=np.array(data).astype(float)
    label=np.array(label).reshape(-1).astype(int)
    # 构建GeoJSON格式数据
    features = []
    for i in range(len(data)):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [data[i][0], data[i][1]]
            },
            "properties": {
                "label": int(label[i])
            }
        }
        features.append(feature)

    geojsonData = {"type": "FeatureCollection", "features": features}
    # 将数据转换为JSON格式，并且不转义非ASCII字符
    json_data_string = json.dumps(geojsonData, ensure_ascii=False)
    return json_data_string

def to_html(args):
    
    # 加载模板文件夹
    env = Environment(loader=FileSystemLoader(os.path.join(script_dir, 'templates')))

    # 选择一个模板文件
    template = env.get_template('template.html')

    # 渲染模板
    rendered_html = template.render(args)
    
    # 构建输出文件路径
    output_path = os.path.join(script_dir,'outputs/process.html')
    
    # 将渲染后的HTML写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered_html)
    final_html = remove_dependencies(output_path)
    os.remove(output_path)
    return final_html
            
def remove_dependencies(inputfilename):
    # 读取 HTML 文件
    with open(inputfilename, "r", encoding="utf-8") as file:
        html_content = file.read()

    # 获取 HTML 文件所在的目录
    html_directory = os.path.dirname(os.path.abspath(inputfilename))

    # 使用BeautifulSoup解析 HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # 提取所有的 CSS 和 JS
    css_code = ""
    js_code = ""

    # 用于保存新的 style 和 script 标签
    new_style_tags = []
    new_script_tags = []
    for link in soup.find_all("link"):
        if link.get("rel") == ["stylesheet"]:
            if link.get("href").startswith("http"):
                continue
            # 获取 CSS 文件的绝对路径
            css_path = os.path.join(html_directory, link["href"])
            # 检查 CSS 文件是否存在
            if os.path.exists(css_path):
                # 如果存在，读取文件内容
                with open(css_path, "r", encoding="utf-8") as file:
                    css_code += file.read() + "\n"
            # 移除<link>标签
            link.extract()

    for script in soup.find_all("script"):
        if script.get("src"):
            if script.get("src").startswith("http"):
                continue
            # 获取 JS 文件的绝对路径
            script_path = os.path.join(html_directory, script["src"])
            # 检查 JS 文件是否存在
            if os.path.exists(script_path):
                # 如果存在，读取文件内容
                with open(script_path, "r", encoding="utf-8") as file:
                    js_code += file.read() + "\n"
            # 移除<script>标签
            script.extract()
        else:
            # 如果没有src属性，直接使用脚本内容
            if script.string:
                js_code += script.string + "\n"
            # 移除<script>标签
            script.extract()

    # 创建新的 style 和 script 标签
    if css_code:
        style_tag = soup.find_all("style")[0]
        style_tag.string = style_tag.string+'\n'+css_code
        new_style_tags.append(style_tag)

    if js_code:
        script_tag = soup.new_tag("script")
        script_tag.string = js_code
        new_script_tags.append(script_tag)

    # 插入新的 style 和 script 标签到原来的位置
    for style_tag in reversed(new_style_tags):
        head_tag = soup.find("head")
        head_tag.append(style_tag)

    for script_tag in reversed(new_script_tags):
        head_tag = soup.find("body")
        head_tag.append(script_tag)
    return str(soup)

def combine_vertical_screenshots(screen_images):
    combined_height = sum([img.height for img in screen_images])
    combined_image = Image.new('RGB', (screen_images[0].width, combined_height), color=(255, 255, 255))  # 创建白色背景的大画布
    offset = 0
    for img in screen_images:
        combined_image.paste(img, (0, offset))
        offset += img.height
    return combined_image

def combine_horizontal_screenshots(column_images):
    combined_width = sum([img.width for img in column_images])
    combined_image = Image.new('RGB', (combined_width, column_images[0].height), color=(255, 255, 255))  # 创建白色背景的大画布
    offset = 0
    for img in column_images:
        combined_image.paste(img, (offset, 0))
        offset += img.width
    return combined_image

def to_img(htmlname,wait_time):

    # 初始化webdriver，这里以Edge为例
    driver = webdriver.Edge()

    # 加载HTML文件或URL
    filename=os.path.abspath(htmlname)
    html_source = 'file://'+ filename # 本地文件
    driver.get(html_source)
    # driver.set_window_size(window_width, window_height)
    driver.maximize_window()
    # 设置全局隐式等待时间（建议使用更精准的显式等待）
    driver.implicitly_wait(wait_time)  # 根据页面加载情况适当调整

    # 假设我们有一个关键元素需要加载完成

    key_element_locator = (By.ID, 'main')  # 替换为你的关键元素ID

    # 使用显式等待确保元素加载完成
    wait = WebDriverWait(driver, wait_time)
    key_element=wait.until(EC.presence_of_element_located(key_element_locator))
    # 获取关键元素的位置和尺寸
    location = key_element.location
    size = key_element.size
    # 获取页面总宽度和总高度
    total_width = location['x']+size['width']
    total_height = location['y']+size['height']

    vertical_scroll_pause_time = 2
    horizontal_scroll_pause_time = 2
    # 初始化一个二维列表来存放每一列的截图
    column_screenshots = []

    # 针对每一列进行截图
    current_left = 0
    while current_left < total_width:
        # 滚动到当前列
        driver.refresh()  # 刷新页面
        wait.until(EC.presence_of_element_located(key_element_locator))  # 确保关键元素重新加载
        driver.execute_script(f"window.scrollTo({current_left}, 0)")
        time.sleep(vertical_scroll_pause_time)
        # 处理当前列的垂直滚动截图
        vertical_screenshots = []
        current_top = 0
        while current_top  < total_height:
            # 滚动到当前行
            driver.refresh()  # 刷新页面
            wait.until(EC.presence_of_element_located(key_element_locator))  # 确保关键元素重新加载
            driver.execute_script(f"window.scrollTo({current_left}, {current_top})")
            time.sleep(horizontal_scroll_pause_time)
            # 截取当前视口屏幕内容
            screenshot = driver.get_screenshot_as_png()
            img = Image.open(BytesIO(screenshot)).resize((driver.execute_script("return window.innerWidth"),driver.execute_script("return window.innerHeight")))
            if (total_height-current_top) < driver.execute_script("return window.innerHeight") and current_top!=0:
                if (total_width-current_left) < driver.execute_script("return window.innerWidth") and current_left!=0:
                    img = img.crop(((driver.execute_script("return window.innerWidth")-(total_width-current_left))
                                    ,(driver.execute_script("return window.innerHeight")-(total_height-current_top)),img.width,img.height))
                else:
                    img = img.crop((0,(driver.execute_script("return window.innerHeight")-(total_height-current_top)),img.width,img.height))
            else:
                if (total_width-current_left) < driver.execute_script("return window.innerWidth") and current_left!=0:
                    img = img.crop(((driver.execute_script("return window.innerWidth")-(total_width-current_left)),0,img.width,img.height))
                else:
                    img = img
            vertical_screenshots.append(img)
            # 移动到下一行
            current_top += driver.execute_script("return window.innerHeight")
        # 合并当前列的竖直截图
        column_image = combine_vertical_screenshots(vertical_screenshots)
        column_screenshots.append(column_image)

        # 移动到下一列
        current_left += driver.execute_script("return window.innerWidth")
    # 合并所有列截图成一个完整的截图
    full_image = combine_horizontal_screenshots(column_screenshots).crop((location['x'], location['y'], location['x'] + size['width'], location['y'] + size['height']))
    # 保存最终的完整截图
    # 关闭WebDriver
    driver.quit()
    return full_image
