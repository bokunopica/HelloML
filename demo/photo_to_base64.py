import base64


if __name__=="__main__":
    pic = open('./img/IMG_20230404_110439.jpg', 'rb')
    store = open('corr_define.txt', 'w')
    pic_b64 = base64.b64encode(pic.read())
    store.write(pic_b64.decode('utf8'))
    store.close()
    pic.close()