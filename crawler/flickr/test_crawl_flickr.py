import flickrapi
import urllib
from PIL import Image
import os
import time

# Flickr api access key 
flickr=flickrapi.FlickrAPI('b33da25022c98281f7566cba4da6177c', 'b2aea8dd70d82f92', cache=True)

keyword = 'hanoi landscape'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=20,           # may be you can try different numbers..
                     sort='relevance')

urls = []
for i, photo in enumerate(photos):
    # print(i)
    time.sleep(0.1)

    url = photo.get('url_c')
    urls.append(url)
    
    # get 50 urls
    if i > 1000:
        break

print(*urls, sep='\n')

# for i in range(len(urls)):
#     if urls[i] is not None:
#         img_path = f'data/{i}.jpg'
#         urllib.request.urlretrieve(urls[i], img_path)
#         image = Image.open(img_path) 
#         image = image.resize((256, 256), Image.Resampling.LANCZOS)
#         image.save(img_path)