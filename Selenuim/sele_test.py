from selenium import webdriver


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
client = webdriver.Chrome(executable_path='D:\MyDownloads\chromedriver.exe',chrome_options=chrome_options)

client.get("https://www.baidu.com/")
client.find_element_by_id('kw').send_keys("gzx")
client.find_element_by_id('su').click()