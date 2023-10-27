# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 08:28:49 2022

@author: Clay Technology World, https://clay-atlas.com/us/blog/2020/11/04/python-en-package-pyinstaller-picture/

Modified by: Jaehoon Jung, PhD, OSU
"""

#-- PyInstaller cannot add a picture to an executable program
#-- To add a picture, convert picture to byte (string) data and load it in the gui code

import base64


def pic2str(file, functionName):
    pic = open(file, 'rb')
    content = '{} = {}\n'.format(functionName, base64.b64encode(pic.read()))
    pic.close()

    #-- This function will create a new .py file named pic2str.py to assign the picture byte data to the variable in the pic2str.py.
    with open('osu_pic2str.py', 'a') as f:
        f.write(content)


if __name__ == '__main__':
    pic2str('osu.jpg', 'osu_logo')
