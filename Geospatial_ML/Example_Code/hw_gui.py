# -*- coding: utf-8 -*-
"""
Create a GUI

@author: Jaehoon Jung, PhD, OSU
"""

import tkinter as tk #-- tkinter comes built in with Python that allows you to create GUI
from tkinter import messagebox
from datetime import datetime, date
import hw_modules as osu
from hw_pic2str import osu_logo #-- Convert picture to byte data, put it in code, and load it
from io import BytesIO
import base64
from PIL import ImageTk, Image
import time
# import sys
# sys.setrecursionlimit(5000)


def licenseMessage():
    messagebox.showwarning("Expired Licenses", "Your license has expired. Please renew it") # messagebox always return something 

#-- define a dummy class to create a data structure of multiple attributes 
class structtype():
    pass

def saveKeyParam():
    pixelsize.set(edit_pixelsize.get()); edit_pixelsize.delete(0, "end"); edit_pixelsize.insert(0, pixelsize.get())  
    voxelsize.set(edit_voxelsize.get()); edit_voxelsize.delete(0, "end"); edit_voxelsize.insert(0, voxelsize.get())  
    
    button_saveKeyParam.config(text="Parameters saved")
    button_saveKeyParam.update()
    time.sleep(1)
    button_saveKeyParam.config(text="Save parameters")
    button_saveKeyParam.update()

def Run():
    
    param = structtype()
    param.check_LAS = check_LAS_var.get()
    param.check_TIF = check_TIF_var.get()
    param.check_extension = radio_PCD_var.get()
    param.cellsize = pixelsize.get()  
    param.voxelsize = voxelsize.get() 
    param.angle_threshold = 10 #-- degrees between 0 - 90. The smaller the value, the more vertical points are filtered out 
    param.intensity_threshold = 0.2 #-- between 0 - 1. The smaller the value, the more pixels are retained
    param.unit = 'meter' # or 'feet' # FIXME: add to GUI parameters 
    
    button_run.config(text="Wait...")
    button_run.update() #-- process the pending task to redraw widget
    
    if datetime.strptime(ExpirationDate,"%Y-%m-%d").date() < date.today(): 
        licenseMessage()
    else:   
        osu.main(param,edit_processingTime,edit_progress) # TODO: main
    
    button_run.config(text="Run")
    button_run.update()
    

# %% setting
Software_version = '0.1 (BETA)'
p_fontSize = ("Arial", 10)
p_borderWidth = 4
p_buttonWidth = 20
p_editWidth = 14 
UpdateDate = '2022-11-22' # FIXME: check before shipping str(date.today())
ExpirationDate = '2022-12-31' # FIXME: check before shipping


main = tk.Tk() #-- create a window widget (place holder)
main.title("CE560_GMS ver " + Software_version)
# main.geometry("600x300") #-- fixed window size
main.configure(bg='white') 

#-- create a widget and put it in the window
#-- label widget is a display box where you can put text or images.
label_name = tk.Label(main, text="Geospatial ML", bg='white', font=("Helvetica 30 bold"), padx=5, pady=2) # check before shipping
#-- put the widget in a grid (row and column)
#-- row and column are relative. They ignore between-values 
label_name.grid(row=0, column=0, columnspan=2, pady=5) # span two columns in the grid

#-- container widget to act as a container to divide complex window layouts 
frame_1 = tk.LabelFrame(main, text="Parameters", font=p_fontSize, padx=15, pady=5)
#-- 'sticky' incidates the sides and corners of the cell to which widget sticks 
frame_1.grid(row=1, column=0, sticky="W", padx=10, pady=0) 
frame_2 = tk.LabelFrame(main, text="Configuration", font=p_fontSize, padx=5, pady=22)
frame_2.grid(row=1, column=1, sticky="W", padx=0, pady=0)
frame_3 = tk.LabelFrame(main, text="Process", font=p_fontSize, padx=15, pady=10)
frame_3.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

# %% Key parameters (frame 1)
label_pixelsize = tk.Label(frame_1, text="pixelsize (m)", font=p_fontSize, padx=5, pady=10)
label_pixelsize.grid(row=0, column=0, sticky="W")
#-- create a input box 
pixelsize=tk.DoubleVar();  #-- float value holder to manipulate the values of tkinter widgets
pixelsize.set(0.05); 
edit_pixelsize = tk.Entry(frame_1, width=p_editWidth, font=p_fontSize, borderwidth=p_borderWidth, fg='red', bg='white')
edit_pixelsize.grid(row=0, column=1) #-- row and column of frame_1. it has its own grid system
edit_pixelsize.insert(0, pixelsize.get()) 

label_voxelsize = tk.Label(frame_1, text="voxelsize (m)", font=p_fontSize, padx=5, pady=10)
label_voxelsize.grid(row=1, column=0, sticky="W") 
voxelsize=tk.DoubleVar(); 
voxelsize.set(0.2); 
edit_voxelsize = tk.Entry(frame_1, width=p_editWidth, font=p_fontSize, borderwidth=p_borderWidth)
edit_voxelsize.grid(row=1, column=1)
edit_voxelsize.insert(0, voxelsize.get()) 

#-- some widgets allow you to run a function using the command option
button_saveKeyParam = tk.Button(frame_1, text="Save parameters", width=p_buttonWidth, height=2, command=saveKeyParam, font=p_fontSize, fg="blue", bg="white")
button_saveKeyParam.grid(row=5, columnspan=2, padx=5, pady=5)

# %% Configuration (frame 2)
#-- checkbox allows to implement on/off switch
check_TIF_var = tk.IntVar()
# check_TIF_var.set(1) #-- check the box by default 
check_TIF = tk.Checkbutton(frame_2, text="Write TIF", font=p_fontSize, variable = check_TIF_var)
check_TIF.grid(row=4, column=0, sticky="W", padx=5, pady=5)

check_LAS_var = tk.IntVar()
check_LAS = tk.Checkbutton(frame_2, text="Write:", font=p_fontSize, variable = check_LAS_var)
check_LAS.grid(row=5, column=0, sticky="W", padx=5, pady=5)

#-- radiobutton widget allows the user to make a selection for only one option from a set of given choices
radio_PCD_var = tk.IntVar()
radio_PCD_var.set(1)
radio_LAS = tk.Radiobutton(frame_2, text="LAS", font=p_fontSize, variable = radio_PCD_var, value=1)
radio_LAS.grid(row=5, column=0, sticky="W", padx=(75,5), pady=5)
radio_LAZ = tk.Radiobutton(frame_2, text="LAZ", font=p_fontSize, variable = radio_PCD_var, value=2)
radio_LAZ.grid(row=5, column=0, sticky="W", padx=(135,5), pady=5)


# %% run the program (frame 3)
button_run_var = tk.StringVar() # example of text value holder 
button_run_var.set("Run")
#-- command option to respond to the button click event 
button_run = tk.Button(frame_3, text=button_run_var.get(), width=p_buttonWidth, height=2, command=Run, font=p_fontSize, bg="white")
button_run.grid(row=0, column=1, padx=5, pady=0)

button_exit = tk.Button(frame_3, text="Exit", width=p_buttonWidth, height=2, command=main.destroy, font=p_fontSize, bg="white")
button_exit.grid(row=0, column=2, padx=5, pady=5)


# %% show progress 
edit_progress = tk.Entry(main, width=50, font=p_fontSize, borderwidth=p_borderWidth, justify='center')
edit_progress.grid(row=3, column=0, columnspan=2, pady=10)
edit_progress.insert(0, "Progress")

edit_processingTime = tk.Entry(main, width=50, font=p_fontSize, borderwidth=p_borderWidth, justify='center')
edit_processingTime.grid(row=4, column=0, columnspan=2, pady=10)
edit_processingTime.insert(0, "Processing Time")

# %% add Software Information 
label_info_1 = tk.Label(main, text="Updated " + UpdateDate + " for version " + Software_version, bg='white', font=p_fontSize, padx=5, pady=2) # check before shipping
label_info_1.grid(row=5, column=0, sticky="W") 
label_info_2 = tk.Label(main, text="Developed by YOUR_NAME, OSU", bg='white', font=p_fontSize, padx=5, pady=2)
label_info_2.grid(row=6, column=0, sticky="W") 
label_info_3 = tk.Label(main, text="YOUR_EMAIL@oregonstate.edu", bg='white', font=p_fontSize, padx=5, pady=2)
label_info_3.grid(row=7, column=0, sticky="W")
label_info_4 = tk.Label(main, text="This is a trial version and will expire on " + ExpirationDate, bg='white', font=p_fontSize, padx=5, pady=2) 
label_info_4.grid(row=8, column=0, columnspan=2) # check before shipping

# %% add logo
osu_logo = BytesIO(base64.b64decode(osu_logo))
img_osu_logo = Image.open(osu_logo)
#-- LANCZOS: interpolation method used to compute new values for sampled data
#-- if you run into "pyimage1 doesn't exist" error, use Image.ANTIALIAS instead of Image.Resampling.LANCZOS
#-- or do not use image.resize 
img_osu_logo = img_osu_logo.resize((200,60), Image.Resampling.LANCZOS)
# img_osu_logo = img_osu_logo.resize((200,60), Image.ANTIALIAS) 
#-- Use the Tkinter PhotoImage widget to display an image for a Label or Button 
img_osu_logo = ImageTk.PhotoImage(img_osu_logo)
tk.Label(main, image=img_osu_logo).grid(row=5, rowspan=3, column=1, pady=5)

# %% 
#-- call mainloop when you are ready for your application to run
#-- mainloop runs until the main window is destoryed 
main.mainloop() 





















