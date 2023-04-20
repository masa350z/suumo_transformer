# %%
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
from tqdm import tqdm
import requests
import random
import os


def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)


def ret_link_list(tokyo_url, chiba_url):
    link_list = []

    for url in [tokyo_url, chiba_url]:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

        nums = soup.find('div', class_='pagination pagination_set-nav')
        num_pages = int(nums.find('ol').find_all('li')[-1].text)

        for page in tqdm(range(num_pages)):
            res = requests.get(url + '&page={}'.format(int(page+1)))
            soup = BeautifulSoup(res.text, 'html.parser')

            bukken_list = soup.find('div', id='js-bukkenList')
            bukken_list = bukken_list.find_all('div', class_='cassetteitem')

            for bukken in bukken_list:
                links = bukken.find('table').find_all('a')[::-2]
                for i in links:
                    link_list.append('https://suumo.jp' + i['href'])

    return link_list


def save_images(soup, chintai_id):
    images = soup.find('ul', id='js-view_gallery-list')
    images = images.find_all('img')
    images = [i['data-src'] for i in images]

    for i in range(len(images)):
        img_name = images[i].split('/')[-1].split('_')[-1]
        path = 'datas/' + chintai_id + '/img/{}'.format(img_name)
        download_file(images[i], path)


def save_texts(soup, chintai_id):
    yachin = soup.find('div', class_='property_view_note-list')
    text0 = ''
    for i in yachin.find_all('span'):
        text0 += i.text.replace('\xa0','')

    path = 'datas/' + chintai_id + '/text/text0.txt'
    with open(path, mode='w') as f:
        f.write(text0)

    table = soup.find('table', class_='property_view_table')
    elements = table.find_all()

    text1 = ''
    for i in [el.text for el in elements]:
        text1 += i + ','

    text2 = soup.find('div', id='bkdt-option').find('li').text

    table = soup.find('table', class_='table_gaiyou')
    elements = table.find_all('tr')

    text3 = ''
    for el in elements:
        el_lis = [i.text for i in el]
        for i in el_lis:
            text3 += i + ','

    path = 'datas/' + chintai_id + '/text/text1.txt'
    with open(path, mode='w') as f:
        f.write(text1)

    path = 'datas/' + chintai_id + '/text/text2.txt'
    with open(path, mode='w') as f:
        f.write(text2)

    path = 'datas/' + chintai_id + '/text/text3.txt'
    with open(path, mode='w') as f:
        f.write(text3)


def save_datas(link):
    chintai_id = link.split('chintai/')[1].split('/')[0]
    if not os.path.exists('datas/' + chintai_id):
        os.makedirs('datas/' + chintai_id)
        os.makedirs('datas/' + chintai_id + '/text')
        os.makedirs('datas/' + chintai_id + '/img')

    res = requests.get(link)
    soup = BeautifulSoup(res.text, 'html.parser')

    save_images(soup, chintai_id)
    save_texts(soup, chintai_id)

    return chintai_id


tokyo_url = 'https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&rn=0580&ek=058019450&ek=058002990&ek=058024780&rn=0540&ek=054000190&rn=0550&rn=0560&ek=056019660&ek=056039760&rn=0025&ek=002537450&ek=002512030&ek=002538520&ek=002531240&ek=002529360&ra=013&ae=05801&ae=00251&ae=05601&cb=4.5&ct=6.5&co=1&md=01&md=02&md=03&md=04&md=05&md=06&ts=1&ts=2&et=9999999&mb=20&mt=9999999&cn=9999999&tc=0400301'
chiba_url = 'https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&rn=0590&ek=059031750&ek=059011400&ek=059039990&ra=012&cb=4.5&ct=6.5&co=1&md=01&md=02&md=03&md=04&md=05&md=06&ts=1&ts=2&et=9999999&mb=20&mt=9999999&cn=9999999&tc=0400301'

link_list = ret_link_list(tokyo_url, chiba_url)
random.shuffle(link_list)

print('あり: 1')
print('ありだけど高い: 2')
print('なし: 3')
for link in link_list:
    chintai_id = link.split('chintai/')[1].split('/')[0]
    if not os.path.exists('datas/' + chintai_id + '/img'):
        chintai_id = save_datas(link)

        print(link)

        inp = 0
        while not (int(inp) == 1 or int(inp) == 2 or int(inp) == 3):
            inp = input('>>')

        path = 'datas/' + chintai_id + '/label.txt'
        with open(path, mode='w') as f:
            f.write(inp)
        
        print('=======================')
# %%
link