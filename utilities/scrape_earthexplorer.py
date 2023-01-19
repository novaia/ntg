from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

username = ''
password = ''

driver = webdriver.Chrome()
driver.implicitly_wait(1000)

# Login.
driver.get('https://ers.cr.usgs.gov/login')

username_input = driver.find_element(By.NAME, 'username')
username_input.send_keys(username)

password_input = driver.find_element(By.NAME, 'password')
password_input.send_keys(password)

login_button = driver.find_element(By.ID, 'loginButton')
login_button.click()

# Navigate to SRTM 1 Arc-Second results.
driver.get('https://earthexplorer.usgs.gov/')

datasets_tab = driver.find_element(By.ID, 'tab2')
datasets_tab.click()
print('Started loading datasets tab.')

digital_elevation_li = driver.find_element(By.ID, 'cat_207')
digital_elevation_expander = digital_elevation_li.find_element(By.CLASS_NAME, 'folder');
digital_elevation_expander.click()

srtm_li = driver.find_element(By.ID, 'cat_1103')
srtm_expander = srtm_li.find_element(By.CLASS_NAME, 'folder')
srtm_expander.click()

one_arcsecond_checkbox = driver.find_element(By.ID, 'coll_5e83a3ee1af480c5')
one_arcsecond_checkbox.click()
print('Selected SRTM 1 arc-second.')

results_tab = driver.find_element(By.ID, 'tab4')
results_tab.click()
print('Started loading results tab.')

# Download SRTM heightmaps.
for k in range(1, 1428):
    page_selector = driver.find_element(By.ID, 'pageSelector_5e83a3ee1af480c5_F')

    if int(page_selector.get_attribute('value')) != k:
        print('Waiting for page ' + str(k) + ' to load.')
        while int(page_selector.get_attribute('value')) != k:
            time.sleep(driver, 0.5)
    print('Page ' + str(k) + ' has loaded.')

    download_options_buttons = driver.find_elements(By.CLASS_NAME, 'download')

    for i in range(len(download_options_buttons)):
        current_result_number = (k - 1) * len(download_options_buttons) + i + 1
        current_result_number_string = str(current_result_number)

        download_options_buttons[i].click()
        print('Opened download menu for result ' + current_result_number_string + '.')

        download_options_container = driver.find_element(By.ID, 'optionsContainer')
        geotiff_download_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[7]/div[2]/div/div[2]/div[3]/div[1]/button')))
        #download_buttons = download_options_container.find_elements(By.CLASS_NAME, 'downloadButtons')
        #geotiff_download_button = download_buttons[2]
        geotiff_download_button.click()
        print('Started downloading result ' + current_result_number_string + '.')

        close_button = driver.find_element(By.XPATH, '/html/body/div[7]/div[1]/button')
        close_button.click()
        print('Closed download menu for result ' + current_result_number_string + '.')
    
    page_selector = driver.find_element(By.ID, 'pageSelector_5e83a3ee1af480c5_F')
    page_selector.send_keys(Keys.DELETE, Keys.DELETE, Keys.DELETE, Keys.DELETE)
    page_selector.send_keys(str(k + 1))
    page_selector.send_keys(Keys.RETURN)
    print('Started loading page ' + str(k + 1) + '.')
    time.sleep(10)